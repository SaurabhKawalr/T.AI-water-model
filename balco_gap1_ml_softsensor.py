# ============================================================
# BALCO GAP1 – TEMP GAP & PASTE TEMP ML SOFT SENSOR (TEST READY)
# ============================================================

import PIconnect as PI
import pandas as pd
import pickle
import time
from pytz import timezone
from OSIsoft.AF.Asset import AFEnumerationValue
from PIconnect.PIConsts import UpdateMode, BufferMode

# ============================================================
# CONFIGURATION
# ============================================================

PI_SERVER_NAME = "10.101.71.18"
LOCAL_TZ = timezone("Asia/Kolkata")

MODEL_PATH = "temp_gap_catboost_model.pkl"

INTERVAL = "10m"
DATA_FREQUENCY_MIN = 10
MAX_GAP_MIN = 60

START_TIME = "12-01-2026 23:59"
END_TIME = "*"

# ---------------- OUTPUT TAGS ----------------
OUTPUT_TEMP_GAP_TAG = "BALCO_GAP1_ML_Target_TEMP_GAP"
OUTPUT_PASTE_TEMP_TAG = "BALCO_GAP1_ML_Target_PASTE_TEMP"

# ---------------- DUMMY TAGS -----------------
DUMMY_TEMP_GAP_TAG = "BALCO_GAP1_ML_Target_DUMMY_1"
DUMMY_PASTE_TEMP_TAG = "BALCO_GAP1_ML_Target_DUMMY_2"

# ---------------- TEST MODE ------------------
TEST_MODE = False
TEST_ITERATIONS = 3
TEST_SLEEP_SEC = 600

# ---------------- INPUT TAGS -----------------
TAG_NAMES= [
    "BALCO_GAP1_AnodeNumber_Advise",
    "BALCO_GAP1_AnodeNumber",
    "BALCO_Current_Shift",
    "BALCO_GAP1_VC1_RUNNING_STATUS",
    #"BALCO_GAP1_VC1_RUNNING_STATUS",
    "BALCO_GAP1_VC2_RUNNING_STATUS",
    "BALCO_GAP1_Air_Pressure",
    "BALCO_GAP1_Ball_Mill_Feed_Rate",
    "BALCO_GAP1_Rejected_Code",
    "BALCO_GAP1_Rejected_D",
    "BALCO_GAP1_Density1",
    "BALCO_GAP1_Rejected_H",
    "BALCO_GAP1_Height1",
    "BALCO_GAP1_Rejected_M",
    "BALCO_GAP1_Speed_Fast_RPM",
    "BALCO_GAP1_Speed_Slow_RPM",
    "BALCO_GAP1_Speed_Fast",
    "BALCO_GAP1_Vibrating_Time",
    "BALCO_GAP1_Transfer_Hopper_Residual_Weight",
    "BALCO_GAP1_Vacuum1",
    "BALCO_GAP1_VC1_Current",
    "BALCO_GAP1_VC2_Current",
    "BALCO_GAP1_VC_REPORT_VC_CURRENT_CALC",
    "BALCO_GAP1_Weight1",
    "BALCO_GAP1_CoolerPower_Calc",
    "BALCO_GAP1_Cooler_Power",
    "BALCO_GAP1_Gate_Position",
    "BALCO_GAP1_Paste_Load",
    "BALCO_GAP1_Paste_Temperature",
    "BALCO_GAP1_Water_Flowrate",
    "BALCO_GAP1_Ball_mill_fraction",
    "BALCO_GAP1_Coarse_Butt",
    "BALCO_GAP1_Coarse_Coke",
    "BALCO_GAP1_Dry_Mix",
    "BALCO_GAP1_Fine_Butt",
    "BALCO_GAP1_Fine_Coke",
    "BALCO_GAP1_Through_PUT",
    "BALCO_GAP1_Green_Scrap",
    "BALCO_GAP1_Medium_Coke",
    "BALCO_GAP1_Running_Pitch",
    "BALCO_GAP1_Set_Pitch",
    "BALCO_GAP1_Total_Paste",
    "BALCO_GAP1_KNEADER_RUNNING_STATUS",
    "BALCO_GAP1_Int_Power",
    "BALCO_GAP1_Kneader_Speed",
    "BALCO_GAP1_Kneader_Torque",
    "BALCO_GAP1_Mixer_Temp",
    "BALCO_GAP1_Pitch_Temp_New",
    "BALCO_GAP1_Pitch_Temp",
    "BALCO_GAP1_Current1",
    "BALCO_GAP1_Dry_Mix_Temp",
    "BALCO_GAP1_Speed1"
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_enum_value(x):
    return x.Name if isinstance(x, AFEnumerationValue) else x


def fetch_pi_data():
    with PI.PIServer(server=PI_SERVER_NAME) as server:
        df_all = pd.DataFrame()

        for tag_name in TAG_NAMES:
            tags = server.search(tag_name)
            if not tags:
                #print(f"⚠️ Tag not found: {tag_name}")
                continue

            tag = tags[0]
            data = tag.interpolated_values(START_TIME, END_TIME, INTERVAL)
            df = data.to_frame(name=tag_name)
            df.index = df.index.tz_convert(LOCAL_TZ)

            df_all = df if df_all.empty else df_all.join(df, how="outer")

    return df_all.map(get_enum_value)


def preprocess_and_interpolate(df):
    df_num = df.apply(pd.to_numeric, errors="coerce")

    max_gap_rows = int(MAX_GAP_MIN / DATA_FREQUENCY_MIN)

    df_interp = df_num.interpolate(
        method="linear",
        limit=max_gap_rows,
        limit_direction="forward",
        axis=0
    )

    missing_cols = df_interp.columns[df_interp.isna().all()]
    df_interp[missing_cols] = 0.0

    numeric_cols = df_interp.select_dtypes(include="number").columns
    df_interp[numeric_cols] = df_interp[numeric_cols].fillna(0).clip(lower=0)

    return df_interp


def load_model():
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    return artifact["model"], artifact["feature_names"]


def apply_downtime_logic(df, pred_series):
    power_low = df["BALCO_GAP1_Int_Power"] < 100

    downtime = power_low.rolling(2, min_periods=2).sum() == 2
    uptime = (~power_low).rolling(1, min_periods=1).sum() == 1

    status = pd.Series("Running", index=df.index)
    status[downtime] = "Downtime"
    status[uptime] = "Running"

    pred_series.loc[status == "Downtime"] = 0
    return pred_series, status


def write_series_to_pi(series, tag_name):
    series = series.dropna().tail(1)

    with PI.PIServer(server=PI_SERVER_NAME) as server:
        tag = server.search(tag_name)[0]

        ts = series.index[0]
        value = float(series.iloc[0])

        tag.update_value(
            value,
            ts,
            UpdateMode.NO_REPLACE,
            BufferMode.BUFFER_IF_POSSIBLE
        )

        #print(f"✅ {tag_name} → {value} @ {ts}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    # print("📥 Fetching PI data...")
    raw_data = fetch_pi_data()

    # print("🧹 Preprocessing data...")
    clean_data = preprocess_and_interpolate(raw_data)

    # print("📦 Loading ML model...")
    model, feature_names = load_model()

    # ========================================================
    # FEATURE ALIGNMENT (THIS FIXES YOUR ERROR)
    # ========================================================

    missing_features = [f for f in feature_names if f not in clean_data.columns]

    if missing_features:
        # print("⚠️ Missing features filled with 0:")
        for f in missing_features:
            #print(f"   - {f}")
            clean_data[f] = 0.0

    # NOW SAFE — NO KeyError POSSIBLE
    X = clean_data[feature_names]

    pred_gap = model.predict(X)

    temp_gap_series = pd.Series(pred_gap, index=clean_data.index)
    paste_pred_series = pd.Series(
        clean_data["BALCO_GAP1_Mixer_Temp"].values - pred_gap,
        index=clean_data.index
    )

    temp_gap_series, plant_status = apply_downtime_logic(
        clean_data, temp_gap_series
    )
    paste_pred_series.loc[plant_status == "Downtime"] = 0

    temp_tag = DUMMY_TEMP_GAP_TAG if TEST_MODE else OUTPUT_TEMP_GAP_TAG
    paste_tag = DUMMY_PASTE_TEMP_TAG if TEST_MODE else OUTPUT_PASTE_TEMP_TAG

    #print("✍ Writing predictions to PI...")
    write_series_to_pi(temp_gap_series, temp_tag)
    write_series_to_pi(paste_pred_series, paste_tag)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    if TEST_MODE:
        #print("🧪 TEST MODE — DUMMY TAGS")
        for i in range(TEST_ITERATIONS):
            #print(f"\n🔁 Iteration {i+1}/{TEST_ITERATIONS}")
            main()
            time.sleep(TEST_SLEEP_SEC)
    else:
        main()
