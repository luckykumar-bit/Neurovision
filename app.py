import streamlit as st
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torchvision import transforms
from PIL import Image
from pathlib import Path
from datetime import datetime
import pandas as pd
import json


# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AgroSNN | Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

IMAGE_SIZE = 128
TIME_STEPS = 25
BETA = 0.9
N_FORWARD_PASSES = 20
VALIDATION_ACCURACY = 70.87

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = (
    Path(__file__).parent / "outputs" / "best_snn.pth"
)


# =========================================================
# 2. CUSTOM CSS - CSS ONLY, NO HTML CONTENT
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

.stApp {
    background-color: #0b1410;
    color: #e8f2eb;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: #101e17;
    border-right: 1px solid #294333;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #e8f5eb !important;
}

p, label {
    color: #c4d8ca;
}

[data-testid="stMetric"] {
    background: #14231a;
    border: 1px solid #2a4232;
    padding: 18px;
    border-radius: 15px;
}

[data-testid="stMetricLabel"] {
    color: #9fbea7 !important;
}

[data-testid="stMetricValue"] {
    color: #e9f8ed !important;
}

.stButton > button {
    background: #23834a;
    color: white;
    border: 1px solid #3d9a60;
    border-radius: 10px;
    min-height: 45px;
    font-weight: 700;
    width: 100%;
}

.stButton > button:hover {
    background: #2c9b59;
    color: white;
    border-color: #5bb87a;
}

.stDownloadButton > button {
    background: #192c20;
    color: white;
    border: 1px solid #3d7650;
    border-radius: 10px;
    min-height: 45px;
    width: 100%;
}

[data-testid="stFileUploader"] {
    background: #14231a;
    border: 1px dashed #477455;
    border-radius: 12px;
    padding: 12px;
}

div[data-testid="stExpander"] {
    background: #14231a;
    border: 1px solid #2a4232;
    border-radius: 12px;
}

hr {
    border-color: #294031;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. SNN MODEL
# IMPORTANT: SAME ARCHITECTURE AS TRAINED CHECKPOINT
# =========================================================

class PlantDiseaseSNN(nn.Module):

    def __init__(self):

        super().__init__()

        sg = surrogate.fast_sigmoid(slope=25)

        self.fc1 = nn.Linear(3, 8)

        self.lif1 = snn.Leaky(
            beta=BETA,
            spike_grad=sg,
            reset_mechanism="subtract"
        )

        self.fc2 = nn.Linear(8, 2)

        self.lif2 = snn.Leaky(
            beta=BETA,
            spike_grad=sg,
            reset_mechanism="subtract"
        )

    def forward(self, x):

        batch_size = x.size(0)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        counts = torch.zeros(
            batch_size,
            2,
            device=x.device
        )

        for _ in range(TIME_STEPS):

            spikes = (
                torch.rand_like(x) < x
            ).float()

            hidden, mem1 = self.lif1(
                self.fc1(spikes),
                mem1
            )

            output, mem2 = self.lif2(
                self.fc2(hidden),
                mem2
            )

            counts += output

        return counts / TIME_STEPS


# =========================================================
# 4. LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model(model_path):

    network = PlantDiseaseSNN().to(DEVICE)

    state = torch.load(
        model_path,
        map_location=DEVICE
    )

    network.load_state_dict(state)

    network.eval()

    return network


if not CHECKPOINT.exists():

    st.error(
        "Model checkpoint not found!"
    )

    st.info(
        "Please place best_snn.pth inside the outputs folder."
    )

    st.stop()


try:

    model = load_model(str(CHECKPOINT))

except Exception as error:

    st.error(f"Model loading error: {error}")

    st.stop()


# =========================================================
# 5. SESSION STATE
# =========================================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0


# =========================================================
# 6. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.title("🌿 AgroSNN")

    st.caption("Plant Disease Intelligence System")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Disease Detection",
            "SNN Architecture",
            "Model Performance",
            "About Project"
        ]
    )

    st.divider()

    st.subheader("System Status")

    st.success("Model Loaded Successfully")

    st.write(
        f"**Processing Device:** {DEVICE}"
    )

    st.write(
        "**Architecture:** 3 → 8 → 2"
    )

    st.write(
        f"**Time Steps:** {TIME_STEPS}"
    )

    st.divider()

    st.caption(
        "FPGA-Based Neuromorphic Plant Disease Detection"
    )

    st.caption(
        "Research Prototype"
    )


# =========================================================
# 7. MAIN HEADER
# =========================================================

st.title("🌿 Plant Disease Intelligence")

st.write(
    "An intelligent plant health classification system "
    "powered by Spiking Neural Networks."
)

st.divider()


# =========================================================
# PAGE 1 - DISEASE DETECTION
# =========================================================

if page == "Disease Detection":

    st.header("Dashboard Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Neural Architecture",
            "3–8–2",
            "Input · Hidden · Output"
        )

    with col2:
        st.metric(
            "Simulation Steps",
            "25",
            "Temporal processing"
        )

    with col3:
        st.metric(
            "Classification",
            "Binary",
            "Healthy / Diseased"
        )

    st.divider()

    st.header("🌱 Leaf Analysis")

    left, right = st.columns(
        [1.1, 0.9],
        gap="large"
    )

    image = None
    uploaded_file = None

    # -----------------------------------------------------
    # IMAGE UPLOAD
    # -----------------------------------------------------

    with left:

        with st.container(border=True):

            st.subheader("Upload Leaf Image")

            st.write(
                "Upload a clear photograph of a plant leaf."
            )

            st.caption(
                "Supported formats: JPG, JPEG, PNG, WEBP"
            )

            uploaded_file = st.file_uploader(
                "Choose a leaf image",
                type=["jpg", "jpeg", "png", "webp"],
                key=f"leaf_{st.session_state.upload_key}"
            )

        if uploaded_file is not None:

            try:

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                st.image(
                    image,
                    caption="Uploaded Plant Leaf",
                    use_container_width=True
                )

                st.caption(
                    f"Dimensions: {image.width} × {image.height}"
                )

            except Exception:

                st.error("Invalid image file.")

                image = None

    # -----------------------------------------------------
    # WORKFLOW
    # -----------------------------------------------------

    with right:

        with st.container(border=True):

            st.subheader("Analysis Workflow")

            st.write("01  Image preprocessing")

            st.write("02  Mean RGB feature extraction")

            st.write("03  Spike-based neural processing")

            st.write("04  LIF neuron simulation")

            st.write("05  Output spike-rate comparison")

        with st.container(border=True):

            st.subheader("Model Information")

            st.write("Input features: 3")

            st.write("Hidden neurons: 8")

            st.write("Output neurons: 2")

            st.write("Neuron model: LIF")

            st.write(f"Time steps: {TIME_STEPS}")

    st.divider()

    # -----------------------------------------------------
    # ACTION BUTTONS
    # -----------------------------------------------------

    button1, button2 = st.columns([2, 1])

    with button1:

        analyze = st.button(
            "🔍 Analyze Plant Health",
            type="primary",
            disabled=image is None,
            use_container_width=True
        )

    with button2:

        reset = st.button(
            "↻ Reset Analysis",
            use_container_width=True
        )

    if reset:

        st.session_state.prediction = None

        st.session_state.upload_key += 1

        st.rerun()

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    if analyze and image is not None:

        transform = transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.ToTensor()
        ])

        x = (
            transform(image)
            .mean(dim=(1, 2))
            .unsqueeze(0)
            .to(DEVICE)
        )

        try:

            with st.spinner(
                "Analyzing plant leaf using SNN..."
            ):

                with torch.no_grad():

                    outputs = torch.stack([
                        model(x)[0]
                        for _ in range(N_FORWARD_PASSES)
                    ]).mean(0)

            healthy_rate = float(
                outputs[0].item()
            )

            diseased_rate = float(
                outputs[1].item()
            )

            prediction = int(
                outputs.argmax().item()
            )

            st.session_state.prediction = {

                "prediction": prediction,

                "label": (
                    "Healthy"
                    if prediction == 0
                    else "Diseased"
                ),

                "healthy_rate": healthy_rate,

                "diseased_rate": diseased_rate,

                "image_name": uploaded_file.name,

                "timestamp": datetime.now().astimezone().isoformat(
                    timespec="seconds"
                )
            }

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )

    # -----------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------

    result = st.session_state.prediction

    if result is not None:

        st.divider()

        st.header("📋 Diagnostic Results")

        if result["prediction"] == 0:

            st.success(
                "🟢 HEALTHY - The Healthy output neuron "
                "produced the higher mean spike rate."
            )

        else:

            st.warning(
                "🟠 DISEASED - The Diseased output neuron "
                "produced the higher mean spike rate."
            )

        # Spike metrics

        r1, r2 = st.columns(2)

        with r1:

            st.metric(
                "Healthy Spike Rate",
                f'{result["healthy_rate"]:.4f}'
            )

        with r2:

            st.metric(
                "Diseased Spike Rate",
                f'{result["diseased_rate"]:.4f}'
            )

        # Chart

        st.subheader("📊 Output Spike Activity")

        chart_data = pd.DataFrame(
            {
                "Mean Spike Rate": [
                    result["healthy_rate"],
                    result["diseased_rate"]
                ]
            },
            index=["Healthy", "Diseased"]
        )

        st.bar_chart(
            chart_data,
            y="Mean Spike Rate",
            height=300
        )

        st.caption(
            "Average output spike rates across 20 stochastic "
            "forward passes. These values are not calibrated probabilities."
        )

        # -------------------------------------------------
        # REPORT
        # -------------------------------------------------

        report = {

            "Project": "AgroSNN",

            "Image": result["image_name"],

            "Prediction": result["label"],

            "Healthy Spike Rate": result["healthy_rate"],

            "Diseased Spike Rate": result["diseased_rate"],

            "Architecture": "3-8-2",

            "Time Steps": TIME_STEPS,

            "Forward Passes": N_FORWARD_PASSES,

            "Validation Accuracy": VALIDATION_ACCURACY,

            "Timestamp": result["timestamp"],

            "Note": (
                "Binary classification research prototype. "
                "Does not identify specific disease types."
            )
        }

        st.divider()

        st.subheader("📥 Export Results")

        d1, d2 = st.columns(2)

        with d1:

            st.download_button(
                "⬇ Download JSON Report",
                data=json.dumps(
                    report,
                    indent=4
                ),
                file_name="AgroSNN_Report.json",
                mime="application/json",
                use_container_width=True
            )

        with d2:

            st.download_button(
                "⬇ Download CSV Data",
                data=chart_data.to_csv().encode("utf-8"),
                file_name="AgroSNN_Spike_Rates.csv",
                mime="text/csv",
                use_container_width=True
            )

        with st.expander(
            "🧠 View SNN Processing Details"
        ):

            st.write("Architecture: 3 → 8 → 2")

            st.write("Neuron: Leaky Integrate-and-Fire")

            st.write(f"Time steps: {TIME_STEPS}")

            st.write(f"Beta: {BETA}")

            st.write(
                f"Forward passes: {N_FORWARD_PASSES}"
            )

            st.write("Input: Mean RGB features")

        with st.expander(
            "⚠️ Research Limitations"
        ):

            st.warning(
                "This prototype performs binary classification "
                "only. It does not identify exact disease types."
            )

            st.write(
                "Mean RGB feature extraction removes spatial "
                "lesion information from the leaf image."
            )

            st.write(
                f"Recorded validation accuracy: "
                f"{VALIDATION_ACCURACY:.2f}%."
            )

            st.write(
                "This is a research prototype, not a validated "
                "agricultural diagnostic system."
            )


# =========================================================
# PAGE 2 - SNN ARCHITECTURE
# =========================================================

elif page == "SNN Architecture":

    st.header("🧠 SNN Architecture")

    st.write(
        "Understanding the neural network structure "
        "used for plant health classification."
    )

    st.divider()

    a, b, c = st.columns(3)

    with a:

        with st.container(border=True):

            st.subheader("Input Layer")

            st.title("3")

            st.caption("Mean RGB features")

    with b:

        with st.container(border=True):

            st.subheader("Hidden Layer")

            st.title("8")

            st.caption("LIF neurons")

    with c:

        with st.container(border=True):

            st.subheader("Output Layer")

            st.title("2")

            st.caption("Healthy / Diseased")

    st.divider()

    st.subheader("Architecture Flow")

    st.code(
        """
        Input Layer
        3 Features (R, G, B)
                |
                v
        Fully Connected Layer
        3 -> 8
                |
                v
        LIF Hidden Neurons
        8 Neurons
                |
                v
        Fully Connected Layer
        8 -> 2
                |
                v
        LIF Output Neurons
        2 Neurons
                |
                v
        Classification
        Healthy / Diseased
        """,
        language="text"
    )

    st.subheader("Neuron Parameters")

    parameters = pd.DataFrame({

        "Parameter": [
            "Neuron Model",
            "Beta",
            "Time Steps",
            "Surrogate Gradient",
            "Reset Mechanism",
            "Input Encoding"
        ],

        "Value": [
            "Leaky Integrate-and-Fire",
            BETA,
            TIME_STEPS,
            "Fast Sigmoid",
            "Subtract",
            "Stochastic Rate Coding"
        ]

    })

    st.dataframe(
        parameters,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# PAGE 3 - MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.header("📈 Model Performance")

    st.write(
        "Summary of the trained Spiking Neural Network."
    )

    p1, p2, p3 = st.columns(3)

    with p1:

        st.metric(
            "Best Validation Accuracy",
            f"{VALIDATION_ACCURACY:.2f}%"
        )

    with p2:

        st.metric(
            "Output Classes",
            "2"
        )

    with p3:

        st.metric(
            "Input Resolution",
            "128 × 128"
        )

    st.divider()

    st.subheader("Training Summary")

    st.write(
        "Dataset: New Plant Diseases Dataset"
    )

    st.write(
        "Training images: 70,295"
    )

    st.write(
        "Validation images: 17,572"
    )

    st.write(
        "Number of classes in original dataset: 38"
    )

    st.write(
        "Current model task: Binary Healthy / Diseased classification"
    )

    st.write(
        "Best recorded validation accuracy: 70.87%"
    )

    st.divider()

    st.info(
        "The validation accuracy is a recorded training result, "
        "not a prediction confidence score. Generalization to "
        "real-world field images has not been established."
    )


# =========================================================
# PAGE 4 - ABOUT PROJECT
# =========================================================

elif page == "About Project":

    st.header("ℹ️ About AgroSNN")

    st.subheader(
        "FPGA-Based Neuromorphic Plant Disease Detection"
    )

    st.write(
        """
        AgroSNN is a research prototype exploring the use
        of Spiking Neural Networks for plant health classification.

        The current software implementation uses a 3-8-2
        feed-forward network with Leaky Integrate-and-Fire neurons.

        The trained model is demonstrated through a
        Streamlit-based software interface.
        """
    )

    st.divider()

    st.subheader("Project Components")

    st.success("Dataset Preparation — Completed")

    st.success("SNN Model Training — Completed")

    st.success("Model Checkpoint Export — Completed")

    st.success("Streamlit Frontend — Implemented")

    st.info("FPGA Hardware Implementation — Pending")

    st.divider()

    st.subheader("Technology Stack")

    st.write("Python")

    st.write("PyTorch")

    st.write("snnTorch")

    st.write("Streamlit")

    st.write("Pandas")

    st.write("Pillow")


# =========================================================
# 8. FOOTER
# =========================================================

st.divider()

st.caption(
    "🌿 AgroSNN | FPGA-Based Neuromorphic Plant Disease Detection"
)

st.caption(
    "Spiking Neural Network · Leaky Integrate-and-Fire · Research Prototype"
)