import datetime
import logging
import os
import pathlib
from dataclasses import dataclass

import holidays
import polars as pl
import pytz
from dotenv import load_dotenv

load_dotenv()

# INPUT_USER_ACTIVITIES_PATH = pathlib.Path("../data/ds-interview-case-tba/DS_interview_case_TBA/user_activities.csv")
# INPUT_USER_ATTRIBUTES_PATH = pathlib.Path("../data/ds-interview-case-tba/DS_interview_case_TBA/user_attributes.csv")
# OUTPUT_PATH = pathlib.Path("../data/processed.parquet")
# HOLIDAYS_PATH = pathlib.Path("../data/holidays.parquet")


@dataclass
class UserActivitesColumns:
    ID: str = "ID"
    WEEK: str = "week"
    YEAR: str = "year"
    START_DATE_OF_WEEK: str = "start_date_of_week"
    UNIQUE_LOTS_VIEWED: str = "unique_lots_viewed"
    AMOUNT_VIEWS: str = "amount_views"
    BIDDED_ON_AMOUNTS_LOTS: str = "bidded_on_amount_lots"
    BIDS_PLACES: str = "bids_places"
    TOTAL_BIDDED: str = "total_bidded"
    MONEY_SPEND: str = "money_spend"
    AMOUNT_LOTS_WON: str = "amount_lots_won"
    PLATFORM: str = "platform"


@dataclass
class UserAttributesColumns:
    ID: str = "ID"
    MOBILE_DEVICE: str = "mobile_device"
    COUNTRY: str = "country"
    PREFERRED_LANGUAGE: str = "preferred_language"
    GENDER: str = "gender"
    IS_COMPANY: str = "is_company"


@dataclass
class ComputedFeatures:
    LAST_ACTIVITY: str = "last_activity"
    INDEX: str = "index"
    END_DATE_OF_WEEK: str = "end_date_of_week"
    CHURNED: str = "churned"
    IS_HOLIDAY: str = "is_holiday"


USER_ACTIVITIES_SCHEMA: dict[str, pl.DataType] = {
    UserActivitesColumns.ID: pl.Int64,
    UserActivitesColumns.WEEK: pl.UInt8,
    UserActivitesColumns.YEAR: pl.UInt16,
    UserActivitesColumns.START_DATE_OF_WEEK: pl.Date,
    UserActivitesColumns.UNIQUE_LOTS_VIEWED: pl.Int32,
    UserActivitesColumns.AMOUNT_VIEWS: pl.Int32,
    UserActivitesColumns.BIDDED_ON_AMOUNTS_LOTS: pl.Int32,
    UserActivitesColumns.BIDS_PLACES: pl.Int32,
    UserActivitesColumns.TOTAL_BIDDED: pl.Float32,
    UserActivitesColumns.MONEY_SPEND: pl.Float32,
    UserActivitesColumns.AMOUNT_LOTS_WON: pl.Float32,
    UserActivitesColumns.PLATFORM: pl.Categorical,
}

numerical_columns: list[str] = [
    UserActivitesColumns.UNIQUE_LOTS_VIEWED,
    UserActivitesColumns.AMOUNT_VIEWS,
    UserActivitesColumns.BIDDED_ON_AMOUNTS_LOTS,
    UserActivitesColumns.BIDS_PLACES,
    UserActivitesColumns.TOTAL_BIDDED,
    UserActivitesColumns.MONEY_SPEND,
    UserActivitesColumns.AMOUNT_LOTS_WON,
]


def create_and_save_holiday_data(
    dataframe: pl.LazyFrame, output_path: pathlib.Path, train: bool = False
) -> pl.LazyFrame:
    country_holidays: dict[str, list[datetime.date]] = {}

    for country in dataframe.select(UserAttributesColumns.COUNTRY).unique().collect().to_series().to_list():
        try:
            country_holidays[country] = list(
                holidays.country_holidays(country=country.upper(), years=(2023, 2024)).keys()
            )
        except NotImplementedError:
            country_holidays[country] = []

    country_holidays: pl.LazyFrame = pl.LazyFrame(
        data=[{"country": key, "holiday": val} for key, values in country_holidays.items() for val in values],
        schema={"country": pl.String, "holiday": pl.Date},
    )

    if train:
        country_holidays.collect().write_parquet(output_path)
    return country_holidays


def create_target_column() -> pl.LazyFrame:
    y: pl.LazyFrame = (
        load_user_activity_data_for_training()
        .with_columns(pl.col(UserActivitesColumns.START_DATE_OF_WEEK).cast(pl.Date))
        .select(UserActivitesColumns.ID, UserActivitesColumns.START_DATE_OF_WEEK)
    )

    y = y.group_by(UserActivitesColumns.ID).agg(pl.col(UserActivitesColumns.START_DATE_OF_WEEK).max())
    max_dataset_date = y.select(pl.col(UserActivitesColumns.START_DATE_OF_WEEK).max()).collect().item()

    return y.with_columns(
        pl.when((max_dataset_date - pl.col(UserActivitesColumns.START_DATE_OF_WEEK)).dt.total_days() > 365)
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias(ComputedFeatures.CHURNED)
    ).drop(UserActivitesColumns.START_DATE_OF_WEEK)


def load_holidays_data(dataframe: pl.LazyFrame, train: bool = False) -> pl.LazyFrame:
    try:
        holidays_path = pathlib.Path(os.getenv("HOLIDAYS_PATH"))
    except TypeError as e:
        logging.error(f"HOLIDAYS_PATH variable not set {e}")
        holidays_path = pathlib.Path("/data/holidays.parquet")

    if holidays_path.exists():
        return pl.scan_parquet(holidays_path)

    return create_and_save_holiday_data(dataframe=dataframe, output_path=holidays_path, train=train)


def add_holidays_data(data: pl.LazyFrame, holidays: pl.LazyFrame) -> pl.LazyFrame:
    data = data.with_row_index(name=ComputedFeatures.INDEX)
    temp_data: pl.LazyFrame = data.select(
        ComputedFeatures.INDEX, UserActivitesColumns.START_DATE_OF_WEEK, UserAttributesColumns.COUNTRY
    )
    temp_data = temp_data.with_columns(
        (pl.col(UserActivitesColumns.START_DATE_OF_WEEK) + pl.duration(days=6)).alias(ComputedFeatures.END_DATE_OF_WEEK)
    )

    temp_data = temp_data.with_columns(pl.col(UserAttributesColumns.COUNTRY).cast(pl.String))
    holidays_data: pl.LazyFrame = holidays.join(other=temp_data, how="left", on=[UserAttributesColumns.COUNTRY])
    holidays_data = holidays_data.sort(ComputedFeatures.INDEX, UserActivitesColumns.START_DATE_OF_WEEK)
    holidays_data = holidays_data.with_columns(
        pl.when(
            pl.col("holiday").is_between(
                pl.col(UserActivitesColumns.START_DATE_OF_WEEK), pl.col(ComputedFeatures.END_DATE_OF_WEEK)
            )
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("is_holiday")
    )
    holidays_data = holidays_data.group_by(ComputedFeatures.INDEX).agg(
        pl.col("is_holiday").sum().alias("holiday_count"),
    )

    return data.join(holidays_data, on=ComputedFeatures.INDEX, how="inner").with_columns(
        pl.col("holiday_count").fill_null(0)
    )


def create_aggregated_features(data: pl.LazyFrame) -> pl.LazyFrame:
    max_dataset_date: datetime.date = datetime.datetime.now(tz=pytz.timezone("Europe/Amsterdam")).today()

    return (
        data.group_by(["ID"])
        .agg(
            pl.when((max_dataset_date - pl.col(UserActivitesColumns.START_DATE_OF_WEEK).last()) > pl.duration(days=365))
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("churned"),
            pl.col(UserActivitesColumns.START_DATE_OF_WEEK).count().alias("num_of_active_weeks"),
            pl.col(UserActivitesColumns.START_DATE_OF_WEEK).max().alias(ComputedFeatures.LAST_ACTIVITY),
            (
                pl.col(UserActivitesColumns.START_DATE_OF_WEEK).max()
                - pl.col(UserActivitesColumns.START_DATE_OF_WEEK).min()
            )
            .dt.total_days()
            .alias("days_subscribed"),
            *[pl.col(col).mean().shrink_dtype().alias(f"{col}_average") for col in numerical_columns],
            *[pl.col(col).min().shrink_dtype().alias(f"{col}_min") for col in numerical_columns],
            *[pl.col(col).max().shrink_dtype().alias(f"{col}_max") for col in numerical_columns],
            *[pl.col(col).median().shrink_dtype().alias(f"{col}_median") for col in numerical_columns],
            *[pl.col(col).sum().shrink_dtype().alias(f"{col}_sum") for col in numerical_columns],
        )
        .sort("ID")
    )


def create_lagged_features(data: pl.LazyFrame, aggregated_data: pl.LazyFrame) -> pl.LazyFrame:
    last_activity_data = data.group_by(UserActivitesColumns.ID).agg(
        pl.col(UserActivitesColumns.START_DATE_OF_WEEK).max().alias(ComputedFeatures.LAST_ACTIVITY)
    )

    data = data.join(
        other=last_activity_data,
        on=UserActivitesColumns.ID,
        how="left",
    ).sort(UserActivitesColumns.ID, UserActivitesColumns.START_DATE_OF_WEEK)

    for days in [30, 90, 180, 365]:
        lagged_data: pl.LazyFrame = (
            data.filter(
                (
                    pl.col(ComputedFeatures.LAST_ACTIVITY) - pl.col(UserActivitesColumns.START_DATE_OF_WEEK)
                ).dt.total_days()
                <= days
            )
            .group_by(UserActivitesColumns.ID)
            .agg(
                *[
                    pl.col(col).mean().shrink_dtype().alias(f"{col}_average_last_{days}_days")
                    for col in numerical_columns
                ],
                *[pl.col(col).min().shrink_dtype().alias(f"{col}_min_last_{days}_days") for col in numerical_columns],
                *[pl.col(col).max().shrink_dtype().alias(f"{col}_max_last_{days}_days") for col in numerical_columns],
                *[
                    pl.col(col).median().shrink_dtype().alias(f"{col}_median_last_{days}_days")
                    for col in numerical_columns
                ],
                *[pl.col(col).sum().shrink_dtype().alias(f"{col}_sum_last_{days}_days") for col in numerical_columns],
                pl.col("holiday_count").sum().shrink_dtype().alias(f"num_of_holidays_in_the_last_{days}_days"),
            )
        )

        aggregated_data = aggregated_data.join(other=lagged_data, on=UserActivitesColumns.ID, how="left")

    return aggregated_data.fill_null(0).drop(ComputedFeatures.LAST_ACTIVITY)


def load_attributes_data_for_training() -> pl.LazyFrame:
    try:
        return pl.scan_csv(os.getenv("USER_ATTRIBUTES_PATH"))
    except FileNotFoundError as e:
        logging.error(f"USER_ATTRIBUTES_PATH variable not set, cannot open file. {e}")
        return pl.LazyFrame()


def load_attributes_data_for_inference(user_id: int) -> pl.LazyFrame:
    try:
        return pl.scan_csv(os.getenv("USER_ATTRIBUTES_PATH")).filter(pl.col(UserAttributesColumns.ID) == user_id)
    except FileNotFoundError as e:
        logging.error(f"USER_ATTRIBUTES_PATH variable not set, cannot open file. {e}")
        return pl.LazyFrame()


def preprocess_attributes_data(attributes_data: pl.LazyFrame) -> pl.LazyFrame:
    return (
        attributes_data.with_columns(
            *[pl.col(col).str.to_lowercase() for col in attributes_data.select(pl.col(pl.String)).columns]
        )
        .with_columns(pl.col(UserAttributesColumns.MOBILE_DEVICE).fill_null("unknown"))
        .with_columns(pl.col(pl.String).cast(pl.Categorical))
    )


def load_user_activity_data_for_inference(user_id: int) -> pl.LazyFrame:
    try:
        return pl.scan_csv(source=os.getenv("USER_ACTIVITIES_PATH"), schema=USER_ACTIVITIES_SCHEMA).filter(
            pl.col(UserActivitesColumns.ID) == user_id
        )
    except FileNotFoundError as e:
        logging.error(e)
        return pl.LazyFrame()


def load_user_activity_data_for_training() -> pl.LazyFrame:
    try:
        return pl.scan_csv(source=os.getenv("USER_ACTIVITIES_PATH"), schema=USER_ACTIVITIES_SCHEMA)
    except FileNotFoundError as e:
        logging.error(e)
        return pl.LazyFrame()


def preprocess_user_activity_data(data: pl.LazyFrame, holidays_data: pl.LazyFrame) -> pl.LazyFrame | None:
    data = add_holidays_data(data=data, holidays=holidays_data)
    aggregated_data = create_aggregated_features(data=data)

    aggregated_data = create_lagged_features(data=data, aggregated_data=aggregated_data)

    return aggregated_data.with_columns(pl.col(pl.String).cast(pl.Categorical))


def load_and_preprocess_data_for_training() -> pl.LazyFrame | None:
    preprocessed_data_path = pathlib.Path(os.getenv("PREPROCESSED_DATA"))
    if preprocessed_data_path.exists():
        return pl.scan_parquet(preprocessed_data_path)

    user_attributes: pl.LazyFrame = load_attributes_data_for_training()
    user_attributes = preprocess_attributes_data(user_attributes)

    data: pl.LazyFrame = load_user_activity_data_for_training()
    data = data.join(other=user_attributes, on=UserAttributesColumns.ID, how="left")

    holidays_data: pl.LazyFrame = load_holidays_data(dataframe=data, train=True)
    data = preprocess_user_activity_data(data, holidays_data=holidays_data).drop(
        [ComputedFeatures.INDEX, ComputedFeatures.CHURNED]
    )

    data = data.join(other=user_attributes, on=UserAttributesColumns.ID, how="left")

    y = create_target_column()
    data = data.join(other=y, on=UserAttributesColumns.ID, how="inner")

    data.collect().write_parquet(preprocessed_data_path)
    return data


def load_and_preprocess_data_for_inference(user_id: int) -> pl.LazyFrame | None:
    user_attributes: pl.LazyFrame = load_attributes_data_for_inference(user_id=user_id)
    user_attributes = preprocess_attributes_data(user_attributes)

    data: pl.LazyFrame = load_user_activity_data_for_inference(user_id=user_id)
    data = data.join(
        other=user_attributes.select(UserAttributesColumns.ID, UserAttributesColumns.COUNTRY),
        on=UserAttributesColumns.ID,
        how="left",
    )

    holidays_data: pl.LazyFrame = load_holidays_data(dataframe=data)
    data = preprocess_user_activity_data(data, holidays_data=holidays_data).drop(
        [ComputedFeatures.INDEX, ComputedFeatures.CHURNED]
    )

    return data.join(other=user_attributes, on=UserAttributesColumns.ID, how="left")
