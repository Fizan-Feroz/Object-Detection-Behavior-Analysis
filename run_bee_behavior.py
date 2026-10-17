# ============================================================
# BEE BEHAVIOR DETECTION
# YOLOv8m + ByteTrack + LSTM
# LSTM WITH SPEED & DISTANCE
# LIVE PREVIEW ONLY
# NO FRAME SKIPPING
# ============================================================

from ultralytics import YOLO

try:
    from keras.models import load_model
except ImportError:
    from tensorflow.keras.models import load_model

import joblib
import cv2
import numpy as np

from collections import defaultdict, deque
from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


YOLO_PATH = (
    BASE_DIR
    / "yolov8"
    / "weights"
    / "best.pt"
)


LSTM_DIR = (
    BASE_DIR
    / "LSTM WITH speed & distance"
)


LSTM_PATH = (
    LSTM_DIR
    / "best_bee_lstm.keras"
)


SCALER_PATH = (
    LSTM_DIR
    / "bee_feature_scaler.pkl"
)


ENCODER_PATH = (
    LSTM_DIR
    / "bee_label_encoder.pkl"
)


VIDEO_PATH = (
    BASE_DIR
    / "videos"
    / "20230711a-fan.mp4"
)


# ============================================================
# 2. SETTINGS
# ============================================================

CONF_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50

IMAGE_SIZE = 640

SEQUENCE_LENGTH = 10

SHOW_PREVIEW = True

# GPU: 0 = first CUDA GPU (RTX 4060), "cpu" = fallback.
# Auto-selects GPU when torch CUDA is available, keeps
# everything else (ByteTrack, LSTM) on CPU.
try:
    import torch as _torch

    DEVICE = 0 if _torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = 0

CACHE_DIR = BASE_DIR / ".cache"


# ============================================================
# 3. LSTM FEATURES
# ============================================================

FEATURE_NAMES = [
    "x",
    "y",
    "width",
    "height",
    "distance",
    "speed",
    "direction_sin",
    "direction_cos",
    "acceleration"
]

FEATURE_COUNT = len(FEATURE_NAMES)


# ============================================================
# 4. BEHAVIOR COLORS
# OpenCV uses BGR
# ============================================================

BEHAVIOR_COLORS = {

    "Fanning":
        (0, 255, 255),       # Yellow

    "Defense":
        (0, 0, 255),         # Red

    "Foraging":
        (0, 255, 0),         # Green

    "Washboarding":
        (255, 0, 0),         # Blue
}


DEFAULT_COLOR = (
    255,
    255,
    255
)


# ============================================================
# 5. PRINT CONFIGURATION
# ============================================================

print()
print("=" * 70)
print("BEE BEHAVIOR DETECTION SYSTEM")
print("YOLOv8m + ByteTrack + LSTM")
print("LSTM WITH SPEED & DISTANCE")
print("=" * 70)


print()
print("=" * 70)
print("CONFIGURATION")
print("=" * 70)

print(
    "YOLO:",
    YOLO_PATH
)

print(
    "LSTM:",
    LSTM_PATH
)

print(
    "Scaler:",
    SCALER_PATH
)

print(
    "Encoder:",
    ENCODER_PATH
)

print(
    "Video:",
    VIDEO_PATH
)

print()
print("Features:")

for i, feature in enumerate(
    FEATURE_NAMES
):

    print(
        f"{i}: {feature}"
    )

print()
print(
    "Sequence length:",
    SEQUENCE_LENGTH
)

print(
    "Frame skipping: NONE"
)

print(
    "Output video saving: DISABLED"
)


# ============================================================
# 6. CHECK FILES
# ============================================================

print()
print("=" * 70)
print("CHECKING REQUIRED FILES")
print("=" * 70)


required_files = {

    "YOLO model":
        YOLO_PATH,

    "LSTM model":
        LSTM_PATH,

    "Scaler":
        SCALER_PATH,

    "Label encoder":
        ENCODER_PATH,

    "Video":
        VIDEO_PATH
}


missing_files = []


for name, path in required_files.items():

    exists = path.exists()

    print(
        f"{name:<20}: "
        f"{'FOUND' if exists else 'MISSING'}"
    )

    if not exists:

        print(
            "   ",
            path
        )

        missing_files.append(
            path
        )


if missing_files:

    raise FileNotFoundError(
        "\nRequired files are missing."
    )


print()
print("All required files found.")


# ============================================================
# 7. LOAD YOLO
# ============================================================

print()
print("=" * 70)
print("LOADING YOLO")
print("=" * 70)


yolo = YOLO(
    str(YOLO_PATH)
)


print(
    "YOLO loaded successfully."
)

print(
    "Classes:",
    yolo.names
)


# ============================================================
# 8. LOAD LSTM
# ============================================================

print()
print("=" * 70)
print("LOADING LSTM")
print("=" * 70)


def load_lstm_compat(model_path):
    """Load .keras model across Keras versions.

    Some models were saved with 'quantization_config' in Dense
    config, which older/newer Keras rejects. Strip it via a
    patched copy kept inside the project (.cache/) on TypeError.
    """
    import shutil
    import zipfile

    try:
        return load_model(
            str(model_path),
            compile=False
        )
    except TypeError as e:
        if "quantization_config" not in str(e):
            raise

        print("Patching LSTM for Keras compat "
              "(stripping quantization_config)...")

        CACHE_DIR.mkdir(exist_ok=True)

        tmp = (
            CACHE_DIR
            / f"patched_{Path(model_path).name}"
        )

        shutil.copy(
            str(model_path),
            str(tmp)
        )

        with zipfile.ZipFile(str(tmp), "a") as z:
            raw = z.read("config.json").decode("utf-8")
            raw = raw.replace(
                ', "quantization_config": null', ""
            )
            raw = raw.replace(
                '"quantization_config": null, ', ""
            )
            raw = raw.replace(
                '"quantization_config": null', ""
            )
            z.writestr("config.json", raw)

        return load_model(
            str(tmp),
            compile=False
        )


lstm = load_lstm_compat(LSTM_PATH)


print(
    "LSTM loaded successfully."
)

print(
    "Input shape:",
    lstm.input_shape
)

print(
    "Output shape:",
    lstm.output_shape
)


# ============================================================
# 9. VERIFY LSTM INPUT
# ============================================================

expected_shape = (
    SEQUENCE_LENGTH,
    FEATURE_COUNT
)


actual_sequence_length = (
    lstm.input_shape[1]
)

actual_feature_count = (
    lstm.input_shape[2]
)


if actual_sequence_length != SEQUENCE_LENGTH:

    raise ValueError(
        f"\nLSTM sequence mismatch.\n"
        f"Expected: {SEQUENCE_LENGTH}\n"
        f"Model: {actual_sequence_length}"
    )


if actual_feature_count != FEATURE_COUNT:

    raise ValueError(
        f"\nLSTM feature mismatch.\n"
        f"Expected: {FEATURE_COUNT}\n"
        f"Model: {actual_feature_count}"
    )


# ============================================================
# 10. LOAD SCALER
# ============================================================

print()
print("Loading scaler...")


scaler = joblib.load(
    str(SCALER_PATH)
)


print(
    "Scaler loaded successfully."
)


# ============================================================
# 11. VERIFY SCALER
# ============================================================

if hasattr(
    scaler,
    "n_features_in_"
):

    if scaler.n_features_in_ != FEATURE_COUNT:

        raise ValueError(
            f"\nScaler feature mismatch.\n"
            f"Expected: {FEATURE_COUNT}\n"
            f"Scaler: {scaler.n_features_in_}"
        )


# ============================================================
# 12. LOAD LABEL ENCODER
# ============================================================

print()
print("Loading label encoder...")


encoder = joblib.load(
    str(ENCODER_PATH)
)


print(
    "Label encoder loaded successfully."
)

print(
    "Classes:",
    encoder.classes_
)


# Pre-resolve index -> class name once.
# Avoids a sklearn inverse_transform call per track per frame.
BEHAVIOR_CLASSES = list(
    encoder.classes_
)


# Wrap the LSTM in a tf.function.
#
# Keras 3 model.predict() re-builds a data adapter and
# re-traces on every call, which costs ~75 ms even for this
# tiny model. A cached tf.function returns bit-identical
# probabilities (verified max abs diff 0.0) roughly 1000x
# faster, taking the pipeline from ~0.5 fps to ~15 fps.
try:
    import tensorflow as _tf

    LSTM_FN = _tf.function(
        lambda t: lstm(t, training=False),
        reduce_retracing=True
    )

    print("LSTM tf.function ready (fast path).")

except Exception as _e:

    LSTM_FN = None

    print(
        "tf.function unavailable, "
        f"falling back to predict(): {_e}"
    )


# ============================================================
# 13. OPEN VIDEO
# ============================================================

print()
print("=" * 70)
print("OPENING VIDEO")
print("=" * 70)


cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)


if not cap.isOpened():

    raise RuntimeError(
        f"Could not open video:\n{VIDEO_PATH}"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)

video_width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

video_height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


print(
    "FPS:",
    fps
)

print(
    "Resolution:",
    f"{video_width} x {video_height}"
)

print(
    "Total frames:",
    total_frames
)


# ============================================================
# 14. TRACK HISTORY
# ============================================================

track_history = defaultdict(
    lambda: deque(
        maxlen=SEQUENCE_LENGTH
    )
)


# Previous position information
previous_data = {}


# ============================================================
# 15. PREDICTION FUNCTION
# ============================================================

def predict_behavior(track_id):

    history = track_history[
        track_id
    ]


    if len(history) < SEQUENCE_LENGTH:

        return None, 0.0


    # Convert sequence to NumPy
    sequence = np.array(
        list(history),
        dtype=np.float32
    )


    # Scale 9 features
    scaled_sequence = scaler.transform(
        sequence
    )


    # LSTM expects:
    # (batch, sequence, features)

    model_input = np.expand_dims(
        scaled_sequence,
        axis=0
    )


    # Predict
    if LSTM_FN is not None:

        probabilities = LSTM_FN(
            _tf.convert_to_tensor(
                model_input,
                dtype=_tf.float32
            )
        )[0].numpy()

    else:

        probabilities = lstm.predict(
            model_input,
            verbose=0
        )[0]


    # Highest probability
    predicted_index = int(
        np.argmax(
            probabilities
        )
    )


    confidence = float(
        probabilities[
            predicted_index
        ]
    )


    # Convert index → behavior
    # (pre-resolved list, same order as encoder.classes_)
    behavior = BEHAVIOR_CLASSES[
        predicted_index
    ]


    return (
        str(behavior),
        confidence
    )


# ============================================================
# 16. MAIN PROCESSING LOOP
# ============================================================

frame_number = 0

total_predictions = 0

unique_track_ids = set()


print()
print("=" * 70)
print("STARTING LIVE INFERENCE")
print("=" * 70)

print()
print("Processing EVERY frame.")
print("Press Q or ESC to stop.")
print()


while True:

    # --------------------------------------------------------
    # READ FRAME
    # --------------------------------------------------------

    success, frame = cap.read()


    if not success:

        print()
        print(
            "End of video reached."
        )

        break


    frame_number += 1


    # ========================================================
    # YOLO + BYTETrack
    # EVERY FRAME
    # ========================================================

    results = yolo.track(

        source=frame,

        persist=True,

        tracker="bytetrack.yaml",

        conf=CONF_THRESHOLD,

        iou=IOU_THRESHOLD,

        imgsz=IMAGE_SIZE,

        device=DEVICE,

        verbose=False
    )


    result = results[0]


    # ========================================================
    # CHECK TRACKS
    # ========================================================

    if (

        result.boxes is not None

        and

        result.boxes.id is not None

    ):

        # ----------------------------------------------------
        # BOXES
        # ----------------------------------------------------

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # TRACK IDS
        # ----------------------------------------------------

        track_ids = (
            result.boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )


        # ----------------------------------------------------
        # YOLO CONFIDENCE
        # ----------------------------------------------------

        yolo_confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )


        # ====================================================
        # PROCESS EACH TRACK
        # ====================================================

        for (

            box,

            track_id,

            yolo_conf

        ) in zip(

            boxes,

            track_ids,

            yolo_confidences

        ):


            unique_track_ids.add(
                int(track_id)
            )


            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            x1, y1, x2, y2 = box


            # ------------------------------------------------
            # CENTER
            # ------------------------------------------------

            center_x = (
                x1 + x2
            ) / 2.0


            center_y = (
                y1 + y2
            ) / 2.0


            # ------------------------------------------------
            # NORMALIZED POSITION
            # ------------------------------------------------

            x = (
                center_x /
                video_width
            )


            y = (
                center_y /
                video_height
            )


            # ------------------------------------------------
            # NORMALIZED SIZE
            # ------------------------------------------------

            box_width = (
                x2 - x1
            ) / video_width


            box_height = (
                y2 - y1
            ) / video_height


            # =================================================
            # MOTION FEATURES
            # =================================================

            previous = previous_data.get(
                int(track_id)
            )


            if previous is None:

                distance = 0.0

                speed = 0.0

                direction_sin = 0.0

                direction_cos = 1.0

                acceleration = 0.0


            else:

                dx = (
                    x -
                    previous["x"]
                )


                dy = (
                    y -
                    previous["y"]
                )


                # --------------------------------------------
                # DISTANCE
                # --------------------------------------------

                distance = float(
                    np.sqrt(
                        dx * dx +
                        dy * dy
                    )
                )


                # --------------------------------------------
                # SPEED
                #
                # Distance per frame converted approximately
                # to distance per second.
                # --------------------------------------------

                speed = (
                    distance * fps
                )


                # --------------------------------------------
                # DIRECTION
                # --------------------------------------------

                if distance > 0:

                    angle = np.arctan2(
                        dy,
                        dx
                    )


                    direction_sin = float(
                        np.sin(angle)
                    )


                    direction_cos = float(
                        np.cos(angle)
                    )


                else:

                    direction_sin = (
                        previous[
                            "direction_sin"
                        ]
                    )


                    direction_cos = (
                        previous[
                            "direction_cos"
                        ]
                    )


                # --------------------------------------------
                # ACCELERATION
                # --------------------------------------------

                acceleration = (
                    speed -
                    previous["speed"]
                )


            # =================================================
            # SAVE CURRENT MOTION
            # =================================================

            previous_data[
                int(track_id)
            ] = {

                "x":
                    x,

                "y":
                    y,

                "speed":
                    speed,

                "direction_sin":
                    direction_sin,

                "direction_cos":
                    direction_cos
            }


            # =================================================
            # 9 FEATURES
            # =================================================

            feature_vector = np.array(

                [

                    x,

                    y,

                    box_width,

                    box_height,

                    distance,

                    speed,

                    direction_sin,

                    direction_cos,

                    acceleration

                ],

                dtype=np.float32
            )


            # =================================================
            # ADD TO TRACK HISTORY
            # =================================================

            track_history[
                int(track_id)
            ].append(
                feature_vector
            )


            # =================================================
            # LSTM PREDICTION
            # =================================================

            behavior = None

            behavior_confidence = 0.0


            if len(
                track_history[
                    int(track_id)
                ]
            ) >= SEQUENCE_LENGTH:


                # --------------------------------------------
                # IMPORTANT:
                # Prediction happens on THIS frame.
                # No last_known_prediction.
                # No frame skipping.
                # --------------------------------------------

                (
                    behavior,

                    behavior_confidence

                ) = predict_behavior(

                    int(track_id)

                )


                total_predictions += 1


            # =================================================
            # SELECT COLOR
            # =================================================

            if behavior is not None:

                color = (
                    BEHAVIOR_COLORS.get(
                        behavior,
                        DEFAULT_COLOR
                    )
                )

            else:

                color = DEFAULT_COLOR


            # =================================================
            # DRAW BOUNDING BOX
            # =================================================

            cv2.rectangle(

                frame,

                (
                    int(x1),
                    int(y1)
                ),

                (
                    int(x2),
                    int(y2)
                ),

                color,

                2

            )


            # =================================================
            # CREATE LABEL
            # =================================================

            if behavior is None:

                label = (

                    f"ID {track_id} | "

                    f"Analyzing..."

                )

            else:

                label = (

                    f"ID {track_id} | "

                    f"{behavior} | "

                    f"{behavior_confidence * 100:.1f}%"

                )


            # =================================================
            # TEXT SETTINGS
            # =================================================

            font = (
                cv2.FONT_HERSHEY_SIMPLEX
            )

            font_scale = 0.55

            thickness = 2


            (
                text_width,
                text_height
            ), baseline = cv2.getTextSize(

                label,

                font,

                font_scale,

                thickness

            )


            # =================================================
            # TEXT POSITION
            # =================================================

            text_x = int(x1)

            text_y = int(
                y1 - 10
            )


            # If there isn't enough room above
            # the bounding box, put text below it.

            if text_y < (
                text_height + 10
            ):

                text_y = int(
                    y2 +
                    text_height +
                    10
                )


            # =================================================
            # BLACK TEXT BACKGROUND
            # =================================================

            background_x1 = (
                text_x - 5
            )

            background_y1 = (
                text_y -
                text_height -
                7
            )

            background_x2 = (
                text_x +
                text_width +
                5
            )

            background_y2 = (
                text_y +
                baseline +
                5
            )


            cv2.rectangle(

                frame,

                (
                    background_x1,
                    background_y1
                ),

                (
                    background_x2,
                    background_y2
                ),

                (
                    0,
                    0,
                    0
                ),

                -1

            )


            # =================================================
            # DRAW LABEL
            # =================================================

            cv2.putText(

                frame,

                label,

                (
                    text_x,
                    text_y
                ),

                font,

                font_scale,

                color,

                thickness,

                cv2.LINE_AA

            )


    # ========================================================
    # DISPLAY INFORMATION
    # ========================================================

    info_text = (

        f"Frame: "
        f"{frame_number}/{total_frames}"
        f" | Tracks: "
        f"{len(unique_track_ids)}"
        f" | Predictions: "
        f"{total_predictions}"

    )


    cv2.rectangle(

        frame,

        (10, 10),

        (
            10 +
            11 * len(info_text),
            45
        ),

        (0, 0, 0),

        -1

    )


    cv2.putText(

        frame,

        info_text,

        (15, 35),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        2,

        cv2.LINE_AA

    )


    # ========================================================
    # SHOW LIVE PREVIEW
    # ========================================================

    if SHOW_PREVIEW:

        cv2.imshow(

            "Bee Behavior - YOLOv8m + ByteTrack + LSTM",

            frame

        )


    # ========================================================
    # KEYBOARD CONTROL
    # ========================================================

    key = (
        cv2.waitKey(1)
        & 0xFF
    )


    if key in (

        ord("q"),

        ord("Q"),

        27

    ):

        print()
        print(
            "Stopped by user."
        )

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("PROCESSING COMPLETE")
print("=" * 70)

print(
    "Frames processed:",
    frame_number
)

print(
    "Unique tracks:",
    len(unique_track_ids)
)

print(
    "LSTM predictions:",
    total_predictions
)

print()
print(
    "Output video: NOT SAVED"
)

print(
    "Frame skipping: NONE"
)

print(
    "Last-known prediction: NOT USED"
)

print("=" * 70)