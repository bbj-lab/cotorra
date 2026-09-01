#!/usr/bin/env python3

"""
generates synthetic clif-like raw tables for end-to-end pipeline tests;
adapted from cocoa's tests/synth.py (github.com/bbj-lab/cocoa) so cotorra's
suite can drive the real cocoa collate -> tokenize -> winnow pipeline without
depending on a sibling checkout
"""

import dataclasses
import datetime
import pathlib

import numpy as np
import polars as pl

BASE = datetime.datetime(2024, 1, 1, 8, 0, 0)
HOUR = datetime.timedelta(hours=1)
DAY = datetime.timedelta(days=1)

# lengths of stay in hours, cycled over hospitalizations; two of the five fall
# short of the 24h default winnowing threshold
LOS_HOURS = (6, 18, 30, 73, 121)

RACES = ("Black or African American", "White", "Asian", "Unknown")
ETHNICITIES = ("Non-Hispanic", "Hispanic")
SEXES = ("Female", "Male")
LANGUAGES = ("English", "Spanish")
ADMIT_TYPES = ("Inpatient", "Acute Care Transfer", "Observation")
DISCHARGES = ("Home", "Expired", "Skilled Nursing Facility", "Hospice")

# codes reachable only from a non-training split, to check that the vocabulary
# is learned on training data alone
TUNING_ONLY_LAB = "quokka marker"
HELD_OUT_ONLY_LAB = "zebra marker"

DTTM = pl.Datetime("us")

SCHEMAS = {
    "clif_patient": {
        "patient_id": pl.String,
        "race_category": pl.String,
        "ethnicity_category": pl.String,
        "sex_category": pl.String,
        "language_category": pl.String,
    },
    "clif_hospitalization": {
        "hospitalization_id": pl.String,
        "patient_id": pl.String,
        "admission_dttm": DTTM,
        "discharge_dttm": DTTM,
        "age_at_admission": pl.Float64,
        "admission_type_category": pl.String,
        "discharge_category": pl.String,
    },
    "clif_adt": {
        "hospitalization_id": pl.String,
        "in_dttm": DTTM,
        "out_dttm": DTTM,
        "location_category": pl.String,
    },
    # deliberately carries no time column: the CODE entry is configured with
    # `reference_key`, so its time arrives from the reference frame
    "clif_code_status": {"patient_id": pl.String, "code_status_category": pl.String},
    "clif_crrt_therapy": {
        "hospitalization_id": pl.String,
        "recorded_dttm": DTTM,
        "crrt_mode_category": pl.String,
        "blood_flow_rate": pl.Float64,
    },
    "clif_labs": {
        "hospitalization_id": pl.String,
        "lab_order_dttm": DTTM,
        "lab_result_dttm": DTTM,
        "lab_category": pl.String,
        "lab_value_numeric": pl.Float64,
    },
    "clif_medication_admin_continuous_converted": {
        "hospitalization_id": pl.String,
        "admin_dttm": DTTM,
        "med_category": pl.String,
        "med_dose_converted": pl.Float64,
        "med_dose_unit_converted": pl.String,
        "_convert_status": pl.String,
    },
    "clif_medication_admin_intermittent_converted": {
        "hospitalization_id": pl.String,
        "admin_dttm": DTTM,
        "med_category": pl.String,
        "med_dose_converted": pl.Float64,
        "med_dose_unit_converted": pl.String,
        "mar_action_category": pl.String,
        "_convert_status": pl.String,
    },
    "clif_patient_assessments": {
        "hospitalization_id": pl.String,
        "recorded_dttm": DTTM,
        "assessment_category": pl.String,
        "numerical_value": pl.Float64,
        "categorical_value": pl.String,
    },
    "clif_position": {
        "hospitalization_id": pl.String,
        "recorded_dttm": DTTM,
        "position_category": pl.String,
    },
    "clif_respiratory_support_processed": {
        "hospitalization_id": pl.String,
        "recorded_dttm": DTTM,
        "mode_category": pl.String,
        "device_category": pl.String,
        "fio2_set": pl.Float64,
        "peep_set": pl.Float64,
        "tidal_volume_set": pl.Float64,
    },
    "clif_sofa": {
        "hospitalization_id": pl.String,
        "event_time": DTTM,
        "sofa_cv_97": pl.Int64,
        "sofa_cns": pl.Int64,
        "sofa_coag": pl.Int64,
        "sofa_liver": pl.Int64,
        "sofa_renal": pl.Int64,
        "sofa_resp": pl.Int64,
        "sofa_total": pl.Int64,
    },
    "clif_vitals": {
        "hospitalization_id": pl.String,
        "recorded_dttm": DTTM,
        "vital_category": pl.String,
        "vital_value": pl.Float64,
    },
    # not referenced by the shipped default config; harmless to generate
    "clif_extra_events": {
        "patient_id": pl.String,
        "event_dttm": DTTM,
        "event_date": pl.Date,
        "extra_category": pl.String,
    },
}


def _spread(idx: int, lo: float, hi: float) -> float:
    """deterministic pseudo-spread of values over [lo, hi]"""
    return round(lo + ((idx * 37 + 11) % 97) / 96 * (hi - lo), 3)


@dataclasses.dataclass(frozen=True)
class Manifest:
    """records what was planted in a generated dataset, for use in assertions"""

    root: pathlib.Path
    tz: str | None
    n_patients: int
    train_frac: float
    tuning_frac: float
    patient_ids: tuple  # in chronological order of first admission
    patient_of: dict  # hospitalization_id -> patient_id
    subjects_of: dict  # patient_id -> tuple of hospitalization_id
    admission: dict  # hospitalization_id -> naive utc admission datetime
    discharge: dict  # hospitalization_id -> naive utc discharge datetime
    los_hours: dict  # hospitalization_id -> length of stay in hours
    icu: frozenset  # hospitalizations with an XFR-IN//icu event
    imv: frozenset  # hospitalizations with a RESP//imv event
    prone: frozenset
    crrt: frozenset
    expired: frozenset
    pressor: frozenset
    tachy: frozenset
    hyperkalemia: frozenset
    nan_vitals: frozenset  # subjects carrying a NaN vital_value
    nan_resp: frozenset  # subjects carrying a NaN peep_set
    inf_resp: frozenset  # subjects carrying an infinite fio2_set

    @property
    def subject_ids(self) -> tuple:
        return tuple(self.patient_of.keys())

    @property
    def split_bounds(self) -> tuple:
        """
        (train, tuning) upper bounds on the chronological patient index;
        mirrors Collator.get_subject_splits, including its float arithmetic --
        np.cumsum([0.7, 0.1]) lands on exactly 0.8 where 0.7 + 0.1 does not
        """
        return tuple(
            (self.n_patients * np.cumsum([self.train_frac, self.tuning_frac])).astype(
                int
            )
        )

    def split_of_patient(self, patient_id: str) -> str:
        i = self.patient_ids.index(patient_id)
        lo, hi = self.split_bounds
        return "train" if i < lo else "tuning" if i < hi else "held_out"

    def split_of_subject(self, hospitalization_id: str) -> str:
        return self.split_of_patient(self.patient_of[hospitalization_id])

    def subjects_in_split(self, split: str) -> tuple:
        return tuple(s for s in self.subject_ids if self.split_of_subject(s) == split)

    @property
    def expected_split_sizes(self) -> dict:
        """number of *patients* expected in each split"""
        lo, hi = self.split_bounds
        return {"train": lo, "tuning": hi - lo, "held_out": self.n_patients - hi}

    def long_stay_subjects(self, hours: float) -> tuple:
        return tuple(s for s, h in self.los_hours.items() if h > hours)

    def short_stay_subjects(self, hours: float) -> tuple:
        return tuple(s for s, h in self.los_hours.items() if h <= hours)


def write_raw_dataset(
    dest: pathlib.Path | str,
    *,
    n_patients: int = 40,
    tz: str | None = None,
    csv_tables: tuple = (),
    train_frac: float = 0.7,
    tuning_frac: float = 0.1,
) -> Manifest:
    """
    write a synthetic clif-like raw dataset covering every table referenced by
    the shipped default collation config;

    times are generated as naive utc instants; when `tz` is given they are
    written as tz-aware values in that zone denoting the same instants, so a
    tz-aware dataset collates to the same instants as its naive counterpart
    """
    root = pathlib.Path(dest).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    rows = {t: [] for t in SCHEMAS}
    patient_ids, patient_of, subjects_of = [], {}, {}
    admission, discharge, los_hours = {}, {}, {}
    (
        icu,
        imv,
        prone,
        crrt,
        expired,
        pressor,
        tachy,
        hyperk,
        nan_vitals,
        nan_resp,
        inf_resp,
    ) = (set(), set(), set(), set(), set(), set(), set(), set(), set(), set(), set())
    lo_idx, hi_idx = (n_patients * np.cumsum([train_frac, tuning_frac])).astype(int)

    for i in range(n_patients):
        pid = f"P{i:04d}"
        patient_ids.append(pid)
        split = "train" if i < lo_idx else "tuning" if i < hi_idx else "held_out"
        rows["clif_patient"].append(
            {
                "patient_id": pid,
                "race_category": RACES[i % len(RACES)],
                "ethnicity_category": ETHNICITIES[i % len(ETHNICITIES)],
                "sex_category": SEXES[i % len(SEXES)],
                "language_category": LANGUAGES[i % len(LANGUAGES)],
            }
        )
        rows["clif_code_status"].append(
            {
                "patient_id": pid,
                "code_status_category": ("Full" if i % 3 else "DNR/DNI"),
            }
        )

        # a fifth of the patients have a second hospitalization; the split is
        # assigned per patient, so both land in the same split
        n_hosp = 2 if i % 5 == 3 else 1
        subjects_of[pid] = tuple(f"H{i:04d}{j}" for j in range(n_hosp))

        for j in range(n_hosp):
            hid = f"H{i:04d}{j}"
            v = i + 3 * j  # variation index for this hospitalization
            patient_of[hid] = pid
            a = BASE + i * 30 * DAY + j * 65 * DAY
            los = LOS_HOURS[v % len(LOS_HOURS)]
            d = a + los * HOUR
            admission[hid], discharge[hid], los_hours[hid] = a, d, los

            dscg = DISCHARGES[v % len(DISCHARGES)]
            if dscg == "Expired":
                expired.add(hid)
            rows["clif_hospitalization"].append(
                {
                    "hospitalization_id": hid,
                    "patient_id": pid,
                    "admission_dttm": a,
                    "discharge_dttm": d,
                    "age_at_admission": _spread(v, 18, 95),
                    "admission_type_category": ADMIT_TYPES[v % len(ADMIT_TYPES)],
                    "discharge_category": dscg,
                }
            )

            # --- transfers: every stay starts in the ed; even patients hit the icu
            locs = ["ed", "icu" if i % 2 == 0 else "ward"]
            if los >= 24:
                locs.append("stepdown")
            if i % 2 == 0:
                icu.add(hid)
            step = los / len(locs)
            for k, loc in enumerate(locs):
                rows["clif_adt"].append(
                    {
                        "hospitalization_id": hid,
                        "in_dttm": a + k * step * HOUR,
                        "out_dttm": a + (k + 1) * step * HOUR,
                        "location_category": loc,
                    }
                )

            # --- vitals every 4h
            for t in range(0, los + 1, 4):
                for c, (lo, hi) in {
                    "heart_rate": (55, 115),
                    "sbp": (95, 175),
                    "dbp": (55, 95),
                    "map": (68, 110),
                    "spo2": (90, 100),
                }.items():
                    rows["clif_vitals"].append(
                        {
                            "hospitalization_id": hid,
                            "recorded_dttm": a + t * HOUR,
                            "vital_category": c,
                            "vital_value": _spread(v + t + len(c), lo, hi),
                        }
                    )
            if v % 3 == 0:  # tachycardia label
                tachy.add(hid)
                rows["clif_vitals"].append(
                    {
                        "hospitalization_id": hid,
                        "recorded_dttm": a + 2 * HOUR,
                        "vital_category": "heart_rate",
                        "vital_value": 141.0,
                    }
                )
            if v % 3 == 1:  # hypotension label
                rows["clif_vitals"].append(
                    {
                        "hospitalization_id": hid,
                        "recorded_dttm": a + 2 * HOUR,
                        "vital_category": "sbp",
                        "vital_value": 84.0,
                    }
                )
            if v % 5 == 0:  # hypertension label
                rows["clif_vitals"].append(
                    {
                        "hospitalization_id": hid,
                        "recorded_dttm": a + 3 * HOUR,
                        "vital_category": "dbp",
                        "vital_value": 124.0,
                    }
                )
            if v == 7:  # a nan numeric value, to be dropped when learning bins
                nan_vitals.add(hid)
                rows["clif_vitals"].append(
                    {
                        "hospitalization_id": hid,
                        "recorded_dttm": a + 5 * HOUR,
                        "vital_category": "heart_rate",
                        "vital_value": float("nan"),
                    }
                )
            if v == 11:  # a null time, to be dropped during collation
                rows["clif_vitals"].append(
                    {
                        "hospitalization_id": hid,
                        "recorded_dttm": None,
                        "vital_category": "heart_rate",
                        "vital_value": 77.0,
                    }
                )

            # --- labs every 12h, ordered an hour before the result
            labs = ["sodium", "potassium", "hemoglobin", "white blood cell"]
            if split == "tuning":
                labs.append(TUNING_ONLY_LAB)
            if split == "held_out":
                labs.append(HELD_OUT_ONLY_LAB)
            for t in range(0, los + 1, 12):
                for c, (lo, hi) in {
                    "sodium": (128, 148),
                    "potassium": (3.0, 5.4),
                    "hemoglobin": (8.0, 15.0),
                    "white blood cell": (3.0, 17.0),
                    TUNING_ONLY_LAB: (0.0, 1.0),
                    HELD_OUT_ONLY_LAB: (0.0, 1.0),
                }.items():
                    if c not in labs:
                        continue
                    rows["clif_labs"].append(
                        {
                            "hospitalization_id": hid,
                            "lab_order_dttm": a + t * HOUR,
                            "lab_result_dttm": a + (t + 1) * HOUR,
                            "lab_category": c,
                            "lab_value_numeric": _spread(v + t + len(c), lo, hi),
                        }
                    )
            for cond, cat, val in (
                (v % 4 == 0, "potassium", 6.9),
                (v % 4 == 1, "potassium", 2.1),
                (v % 4 == 2, "hemoglobin", 6.2),
                (v % 7 == 3, "sodium", 165.0),
                (v % 7 == 5, "sodium", 112.0),
            ):
                if cond:
                    if cat == "potassium" and val > 6.5:
                        hyperk.add(hid)
                    rows["clif_labs"].append(
                        {
                            "hospitalization_id": hid,
                            "lab_order_dttm": a + 2 * HOUR,
                            "lab_result_dttm": a + 3 * HOUR,
                            "lab_category": cat,
                            "lab_value_numeric": val,
                        }
                    )

            # --- continuous medications every 6h; every fourth row fails
            # unit conversion and so carries no numeric value
            meds = ["propofol"] + (["norepinephrine"] if v % 3 == 1 else [])
            if v % 3 == 1:
                pressor.add(hid)
            for n, t in enumerate(range(0, los + 1, 6)):
                for c in meds:
                    ok = (n + len(c)) % 4 != 0
                    rows["clif_medication_admin_continuous_converted"].append(
                        {
                            "hospitalization_id": hid,
                            "admin_dttm": a + t * HOUR,
                            "med_category": c,
                            "med_dose_converted": _spread(v + t, 0.5, 30.0),
                            "med_dose_unit_converted": "mcg/kg/min",
                            "_convert_status": (
                                "success"
                                if ok
                                else "original unit dose is not recognized"
                            ),
                        }
                    )

            # --- intermittent medications every 8h; a third are not given
            for n, t in enumerate(range(0, los + 1, 8)):
                for c in ("morphine", "sodium bicarbonate"):
                    rows["clif_medication_admin_intermittent_converted"].append(
                        {
                            "hospitalization_id": hid,
                            "admin_dttm": a + t * HOUR,
                            "med_category": c,
                            "med_dose_converted": _spread(v + t + len(c), 0.5, 50.0),
                            "med_dose_unit_converted": "mg",
                            "mar_action_category": (
                                "given" if (n + len(c)) % 3 else "held"
                            ),
                            "_convert_status": (
                                "success"
                                if (n + len(c)) % 5
                                else "user-preferred unit is not recognized"
                            ),
                        }
                    )

            # --- assessments: rass is quantitative, cam_total qualitative
            for t in range(0, los + 1, 8):
                rows["clif_patient_assessments"].append(
                    {
                        "hospitalization_id": hid,
                        "recorded_dttm": a + t * HOUR,
                        "assessment_category": "rass",
                        "numerical_value": float(-((v + t) % 6) + 1),
                        "categorical_value": None,
                    }
                )
            for t in range(0, los + 1, 12):
                rows["clif_patient_assessments"].append(
                    {
                        "hospitalization_id": hid,
                        "recorded_dttm": a + t * HOUR,
                        "assessment_category": "cam_total",
                        "numerical_value": None,
                        "categorical_value": (
                            "Positive" if (v + t) % 3 == 0 else "Negative"
                        ),
                    }
                )
            if v == 13:  # a null code, to be dropped during collation
                rows["clif_patient_assessments"].append(
                    {
                        "hospitalization_id": hid,
                        "recorded_dttm": a + HOUR,
                        "assessment_category": None,
                        "numerical_value": 3.0,
                        "categorical_value": None,
                    }
                )

            # --- position: only prone is collated
            for n, t in enumerate((1, 5)):
                if t > los:
                    continue
                p = "prone" if v % 6 == 0 and n == 0 else "supine"
                if p == "prone":
                    prone.add(hid)
                rows["clif_position"].append(
                    {
                        "hospitalization_id": hid,
                        "recorded_dttm": a + t * HOUR,
                        "position_category": p,
                    }
                )

            # --- respiratory support every 6h
            vented = v % 3 != 2
            if vented:
                imv.add(hid)
            if vented and v == 6:
                inf_resp.add(hid)
            if vented and v == 9:
                nan_resp.add(hid)
            for n, t in enumerate(range(0, los + 1, 6)):
                rows["clif_respiratory_support_processed"].append(
                    {
                        "hospitalization_id": hid,
                        "recorded_dttm": a + t * HOUR,
                        "mode_category": (
                            "assist control-volume control" if vented else None
                        ),
                        "device_category": "imv" if vented else "nasal cannula",
                        "fio2_set": (
                            float("inf")
                            if (vented and v == 6 and n == 0)
                            else _spread(v + t, 0.21, 1.0)
                            if vented
                            else None
                        ),
                        "peep_set": (
                            float("nan")
                            if (vented and v == 9 and n == 0)
                            else _spread(v + t + 1, 5.0, 18.0)
                            if vented
                            else None
                        ),
                        "tidal_volume_set": (
                            _spread(v + t + 2, 320.0, 620.0) if vented else None
                        ),
                    }
                )

            # --- sofa daily; totals of 2 or more trigger the sepsis label
            for t in range(0, los + 1, 24):
                comps = {
                    "sofa_cv_97": (v + t) % 3,
                    "sofa_cns": (v + t + 1) % 3,
                    "sofa_coag": (v + t + 2) % 2,
                    "sofa_liver": (v + t) % 2,
                    "sofa_renal": (v + t + 1) % 2,
                    "sofa_resp": (v + t + 2) % 3,
                }
                rows["clif_sofa"].append(
                    {
                        "hospitalization_id": hid,
                        "event_time": a + t * HOUR,
                        **comps,
                        "sofa_total": sum(comps.values()),
                    }
                )

            # --- crrt: a null mode is always present and always filtered out
            rows["clif_crrt_therapy"].append(
                {
                    "hospitalization_id": hid,
                    "recorded_dttm": a + HOUR,
                    "crrt_mode_category": None,
                    "blood_flow_rate": 150.0,
                }
            )
            if v % 9 == 4:
                crrt.add(hid)
                for t in (4, 10, 16):
                    if t > los:
                        continue
                    rows["clif_crrt_therapy"].append(
                        {
                            "hospitalization_id": hid,
                            "recorded_dttm": a + t * HOUR,
                            "crrt_mode_category": "CVVHDF",
                            "blood_flow_rate": _spread(v + t, 80.0, 300.0),
                        }
                    )

        # --- patient-level extra events: one inside the first stay, one well
        # before it; the date-only column exercises fix_date_to_time
        first = f"H{i:04d}0"
        rows["clif_extra_events"].extend(
            [
                {
                    "patient_id": pid,
                    "event_dttm": admission[first] + HOUR,
                    "event_date": (admission[first] + HOUR).date(),
                    "extra_category": "inside window",
                },
                {
                    "patient_id": pid,
                    "event_dttm": admission[first] - 10 * DAY,
                    "event_date": (admission[first] - 10 * DAY).date(),
                    "extra_category": "outside window",
                },
            ]
        )

    for table, schema in SCHEMAS.items():
        df = pl.DataFrame(rows[table], schema=schema, orient="row")
        if tz is not None:
            df = df.with_columns(
                pl.col(c).dt.replace_time_zone("UTC").dt.convert_time_zone(tz).alias(c)
                for c, t in schema.items()
                if t == DTTM
            )
        if table in csv_tables:
            df.write_csv(root / f"{table}.csv")
        else:
            df.write_parquet(root / f"{table}.parquet")

    return Manifest(
        root=root,
        tz=tz,
        n_patients=n_patients,
        train_frac=train_frac,
        tuning_frac=tuning_frac,
        patient_ids=tuple(patient_ids),
        patient_of=patient_of,
        subjects_of=subjects_of,
        admission=admission,
        discharge=discharge,
        los_hours=los_hours,
        icu=frozenset(icu),
        imv=frozenset(imv),
        prone=frozenset(prone),
        crrt=frozenset(crrt),
        expired=frozenset(expired),
        pressor=frozenset(pressor),
        tachy=frozenset(tachy),
        hyperkalemia=frozenset(hyperk),
        nan_vitals=frozenset(nan_vitals),
        nan_resp=frozenset(nan_resp),
        inf_resp=frozenset(inf_resp),
    )


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        m = write_raw_dataset(tmp, n_patients=120)
        for f in sorted(pathlib.Path(tmp).glob("*.parquet")):
            print(f"{f.name:52s} {pl.read_parquet(f).shape}")
        print(m.expected_split_sizes)
        print(f"{len(m.subject_ids)} subjects, {m.n_patients} patients")
