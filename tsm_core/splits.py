from __future__ import annotations

from itertools import combinations
from dataclasses import asdict, dataclass
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class FoldSpec:
    fold_id: int
    train_start_idx: int
    train_end_idx: int
    validation_start_idx: int
    validation_end_idx: int
    test_start_idx: int
    test_end_idx: int
    horizon_days: int
    embargo_days: int
    purged_train_count: int = 0
    train_event_count: int = 0
    validation_event_count: int = 0
    test_event_count: int = 0

    def to_row(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CombinatorialFoldSpec:
    path_id: int
    train_group_ids: tuple[int, ...]
    test_group_ids: tuple[int, ...]
    horizon_days: int
    embargo_days: int
    purged_train_count: int = 0
    train_event_count: int = 0
    test_event_count: int = 0

    def to_row(self) -> dict:
        out = asdict(self)
        out["train_group_ids"] = "|".join(str(v) for v in self.train_group_ids)
        out["test_group_ids"] = "|".join(str(v) for v in self.test_group_ids)
        return out


class PurgedEventTimeSplit:
    def __init__(
        self,
        horizon_days: int,
        train_days: int = 756,
        validation_days: int = 40,
        test_days: int = 252,
        step_days: int = 252,
        embargo_days: int | None = None,
        min_train_events: int = 120,
        min_validation_events: int = 40,
        min_test_events: int = 40,
    ) -> None:
        self.horizon_days = int(horizon_days)
        self.train_days = int(train_days)
        self.validation_days = int(validation_days)
        self.test_days = int(test_days)
        self.step_days = int(step_days)
        self.embargo_days = max(int(embargo_days if embargo_days is not None else horizon_days), int(horizon_days))
        self.min_train_events = int(min_train_events)
        self.min_validation_events = int(min_validation_events)
        self.min_test_events = int(min_test_events)

    def iter_specs(self, max_signal_idx: int) -> Iterator[FoldSpec]:
        train_end = self.train_days - 1
        fold_id = 1
        while True:
            validation_start = train_end + 1
            validation_end = validation_start + self.validation_days - 1
            test_start = validation_end + self.embargo_days + 1
            test_end = min(test_start + self.test_days - 1, int(max_signal_idx))
            if test_start > int(max_signal_idx):
                break
            yield FoldSpec(
                fold_id=fold_id,
                train_start_idx=0,
                train_end_idx=train_end,
                validation_start_idx=validation_start,
                validation_end_idx=validation_end,
                test_start_idx=test_start,
                test_end_idx=test_end,
                horizon_days=self.horizon_days,
                embargo_days=self.embargo_days,
            )
            fold_id += 1
            train_end += self.step_days

    @staticmethod
    def purge_train_against_eval(
        train: pd.DataFrame,
        evaluation: pd.DataFrame,
        horizon_days: int,
        embargo_days: int,
        signal_idx_col: str = "signal_idx",
    ) -> pd.DataFrame:
        if train.empty or evaluation.empty:
            return train.copy()
        train_signal = pd.to_numeric(train[signal_idx_col], errors="coerce")
        eval_signal = pd.to_numeric(evaluation[signal_idx_col], errors="coerce").dropna()
        if eval_signal.empty:
            return train.copy()
        eval_start = int(eval_signal.min())
        eval_end = int(eval_signal.max())
        train_label_start = train_signal + 1
        train_label_end = train_signal + int(horizon_days)
        eval_label_start = eval_start + 1
        eval_label_end = eval_end + int(horizon_days)
        overlaps_eval = (train_label_start <= eval_label_end) & (train_label_end >= eval_label_start)
        in_embargo = train_signal >= eval_start - int(embargo_days)
        keep = ~(overlaps_eval | in_embargo)
        return train.loc[keep].copy()

    def apply(self, frame: pd.DataFrame, spec: FoldSpec, signal_idx_col: str = "signal_idx") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FoldSpec]:
        values = pd.to_numeric(frame[signal_idx_col], errors="coerce")
        raw_train = frame[(values >= spec.train_start_idx) & (values <= spec.train_end_idx)].copy()
        validation = frame[(values >= spec.validation_start_idx) & (values <= spec.validation_end_idx)].copy()
        test = frame[(values >= spec.test_start_idx) & (values <= spec.test_end_idx)].copy()

        train_after_validation_purge = self.purge_train_against_eval(
            raw_train,
            validation,
            horizon_days=self.horizon_days,
            embargo_days=self.embargo_days,
            signal_idx_col=signal_idx_col,
        )
        train = self.purge_train_against_eval(
            train_after_validation_purge,
            test,
            horizon_days=self.horizon_days,
            embargo_days=self.embargo_days,
            signal_idx_col=signal_idx_col,
        )
        purged = int(len(raw_train) - len(train))
        out_spec = FoldSpec(
            fold_id=spec.fold_id,
            train_start_idx=spec.train_start_idx,
            train_end_idx=spec.train_end_idx,
            validation_start_idx=spec.validation_start_idx,
            validation_end_idx=spec.validation_end_idx,
            test_start_idx=spec.test_start_idx,
            test_end_idx=spec.test_end_idx,
            horizon_days=spec.horizon_days,
            embargo_days=spec.embargo_days,
            purged_train_count=purged,
            train_event_count=len(train),
            validation_event_count=len(validation),
            test_event_count=len(test),
        )
        return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True), out_spec

    def split(self, frame: pd.DataFrame, signal_idx_col: str = "signal_idx") -> Iterator[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FoldSpec]]:
        max_signal_idx = int(pd.to_numeric(frame[signal_idx_col], errors="coerce").max())
        for spec in self.iter_specs(max_signal_idx):
            train, validation, test, out_spec = self.apply(frame, spec, signal_idx_col=signal_idx_col)
            if len(train) < self.min_train_events or len(validation) < self.min_validation_events or len(test) < self.min_test_events:
                continue
            yield train, validation, test, out_spec


class CombinatorialPurgedEventSplit:
    def __init__(
        self,
        horizon_days: int,
        n_groups: int = 6,
        test_group_count: int = 2,
        embargo_days: int | None = None,
        min_train_events: int = 120,
        min_test_events: int = 40,
        max_paths: int | None = None,
    ) -> None:
        if n_groups < 3:
            raise ValueError("n_groups must be at least 3")
        if test_group_count <= 0 or test_group_count >= n_groups:
            raise ValueError("test_group_count must be between 1 and n_groups - 1")
        self.horizon_days = int(horizon_days)
        self.n_groups = int(n_groups)
        self.test_group_count = int(test_group_count)
        self.embargo_days = max(int(embargo_days if embargo_days is not None else horizon_days), int(horizon_days))
        self.min_train_events = int(min_train_events)
        self.min_test_events = int(min_test_events)
        self.max_paths = int(max_paths) if max_paths is not None else None

    def assign_groups(self, frame: pd.DataFrame, signal_idx_col: str = "signal_idx") -> pd.Series:
        if frame.empty:
            return pd.Series(dtype="int64")
        values = pd.to_numeric(frame[signal_idx_col], errors="coerce")
        order = values.rank(method="first").fillna(values.notna().sum() + 1).astype(int) - 1
        n = max(int(len(frame)), 1)
        groups = (order * self.n_groups // n).clip(lower=0, upper=self.n_groups - 1).astype(int)
        return groups

    def split(self, frame: pd.DataFrame, signal_idx_col: str = "signal_idx") -> Iterator[tuple[pd.DataFrame, pd.DataFrame, CombinatorialFoldSpec]]:
        if frame.empty:
            return
        work = frame.copy()
        work["_cpcv_group_id"] = self.assign_groups(work, signal_idx_col=signal_idx_col).to_numpy()
        group_ids = tuple(range(self.n_groups))
        path_id = 1
        for test_groups in combinations(group_ids, self.test_group_count):
            test_group_set = set(test_groups)
            train_groups = tuple(group_id for group_id in group_ids if group_id not in test_group_set)
            raw_train = work[~work["_cpcv_group_id"].isin(test_group_set)].drop(columns=["_cpcv_group_id"]).copy()
            test = work[work["_cpcv_group_id"].isin(test_group_set)].drop(columns=["_cpcv_group_id"]).copy()
            train = PurgedEventTimeSplit.purge_train_against_eval(
                raw_train,
                test,
                horizon_days=self.horizon_days,
                embargo_days=self.embargo_days,
                signal_idx_col=signal_idx_col,
            )
            if len(train) < self.min_train_events or len(test) < self.min_test_events:
                continue
            spec = CombinatorialFoldSpec(
                path_id=path_id,
                train_group_ids=train_groups,
                test_group_ids=tuple(test_groups),
                horizon_days=self.horizon_days,
                embargo_days=self.embargo_days,
                purged_train_count=int(len(raw_train) - len(train)),
                train_event_count=int(len(train)),
                test_event_count=int(len(test)),
            )
            yield train.reset_index(drop=True), test.reset_index(drop=True), spec
            path_id += 1
            if self.max_paths is not None and path_id > self.max_paths:
                break
