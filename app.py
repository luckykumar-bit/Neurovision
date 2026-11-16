
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torchvision import transforms
from PIL import Image


# =========================================================
# CONFIGURATION
# =========================================================

IMAGE_SIZE = 128
TIME_STEPS = 25
BETA = 0.9
N_FORWARD_PASSES = 20
VALIDATION_ACCURACY = 70.87

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="AgroSNN | Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0b1410;
    color: #e8f2eb;
}

[data-testid="stSidebar"] {
    background: #101e17;
    border-right: 1px solid #294333;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #c4d8ca;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(120deg, #142c20, #1d5033);
    border: 1px solid #376347;
    border-radius: 22px;
    padding: 32px;
    margin-bottom: 28px;
}

.hero-label {
    color: #8bd4a0;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
}

.hero h1 {
    color: #f2fff4;
    font-size: 38px;
    font-weight: 800;
    margin: 12px 0;
}

.hero p {
    color: #c1d9c7;
    font-size: 15px;
    margin-bottom: 0;
}

.section-title {
    font-size: 21px;
    font-weight: 800;
    color: #e9f6ec;
    margin: 20px 0 15px 0;
}

.info-card {
    background: #14231a;
    border: 1px solid #2a4232;
    border-radius: 16px;
    padding: 20px;
    min-height: 115px;
}

.info-label {
    color: #91b49b;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.info-value {
    color: #e8f8ec;
    font-size: 27px;
    font-weight: 800;
    margin-top: 9px;
}

.info-sub {
    color: #8fae98;
    font-size: 12px;
    margin-top: 5px;
}

.panel {
    background: #14231a;
    border: 1px solid #2a4232;
    border-radius: 17px;
    padding: 22px;
    margin-bottom: 15px;
}

.panel h3 {
    color: #e9f6ec;
    font-size: 19px;
}

.muted {
    color: #a4bdaa;
    font-size: 13px;
    line-height: 1.7;
}

.result-card {
    border-radius: 18px;
    padding: 25px;
    margin: 18px 0;
}

.healthy {
    background: linear-gradient(120deg, #153a28, #1b4a31);
    border: 1px solid #347d50;
}

.diseased {
    background: linear-gradient(120deg, #40231f, #512b25);
    border: 1px solid #a95342;
}

.result-label {
    color: #c4ddca;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.5px;
}

.result-value {
    color: white;
    font-size: 31px;
    font-weight: 800;
    margin-top: 9px;
}

.result-description {
    color: #d0dfd4;
    font-size: 14px;
    margin-top: 8px;
}

div.stButton > button,
div.stDownloadButton > button {
    border-radius: 11px;
    min-height: 46px;
    font-weight: 700;
    border: 1px solid #3d7650;
}

div.stButton > button[kind="primary"] {
    background: #2f9b5a;
    color: white;
}

div.stButton > button[kind="primary"]:hover {
    background: #267d49;
    color: white;
}

div.stButton > button[kind="secondary"],
div.stDownloadButton > button {
    background: #192c20;
    color: #e0eee3;
}

div[data-testid="stFileUploader"] {
    background: #14231a;
    border: 1px dashed #477455;
    border-radius: 14px;
    padding: 12px;
}

div[data-testid="stMetric"] {
    background: #14231a;
    border: 1px solid #2a4232;
    border-radius: 14px;
    padding: 15px;
}

div[data-testid="stMetricLabel"] {
    color: #9fbea7;
}

div[data-testid="stMetricValue"] {
    color: #e9f8ed;
}

.footer {
    text-align: center;
    color: #789481;
    font-size: 12px;
    padding-top: 25px;
}

hr {
    border-color: #294031;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SNN MODEL
# Keep this architecture compatible with best_snn.pth
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
            batch_size, 2, device=x.device
        )

        for _ in range(TIME_STEPS):

            spikes = (torch.rand_like(x) < x).float()

            hidden, mem1 = self.lif1(
                self.fc1(spikes), mem1
            )

            output, mem2 = self.lif2(
                self.fc2(hidden), mem2
            )

            counts += output

        return counts / TIME_STEPS


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

CHECKPOINT = (
    Path(__file__).parent / "outputs" / "best_snn.pth"
)


@st.cache_resource
def load_model(path):

    network = PlantDiseaseSNN().to(DEVICE)

    state = torch.load(
        path,
        map_location=DEVICE
    )

    network.load_state_dict(state)
    network.eval()

    return network


if not CHECKPOINT.exists():

    st.error(
        "Model not found: outputs/best_snn.pth"
    )

    st.info(
        "Place best_snn.pth inside the outputs folder."
    )

    st.stop()


try:

    model = load_model(str(CHECKPOINT))

except Exception as error:

    st.error(f"Model loading failed: {error}")
    st.stop()


# =========================================================
# SESSION STATE
# =========================================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "upload_version" not in st.session_state:
    st.session_state.upload_version = 0


def reset_analysis():

    st.session_state.prediction = None

    st.session_state.upload_version += 1


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🌿 AgroSNN")

    st.caption("Plant Disease Intelligence")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Disease Detection",
            "SNN Architecture",
            "Model Performance",
            "About Project"
        ]
    )

    st.divider()

    st.markdown("### System Status")

    st.success("Model loaded successfully")

    st.info(f"Device: {DEVICE}")

    st.divider()

    st.markdown("### Model Configuration")

    st.code("3 → 8 → 2", language="text")

    st.write("Neuron: Leaky Integrate-and-Fire")

    st.write(f"Time steps: {TIME_STEPS}")

    st.write(f"Beta: {BETA}")

    st.divider()

    st.caption(
        "FPGA-Based Neuromorphic Plant Disease Detection"
    )

    st.caption("Research Prototype")


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown("""
<div class="hero">

    <div class="hero-label">
        INTELLIGENT AGRICULTURE · NEUROMORPHIC AI
    </div>

    <h1>🌿 Plant Disease Intelligence</h1>

    <p>
        An FPGA-oriented Spiking Neural Network prototype
        for intelligent plant health classification.
    </p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# PAGE 1: DISEASE DETECTION
# =========================================================

if page == "Disease Detection":

    st.markdown(
        '<div class="section-title">Dashboard Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    metrics = [
        ("Neural Architecture", "3–8–2", "Input · Hidden · Output"),
        ("Simulation Steps", "25", "Temporal spike simulation"),
        ("Classification", "Binary", "Healthy vs Diseased")
    ]

    for col, (label, value, sub) in zip(
        [col1, col2, col3], metrics
    ):

        with col:

            st.markdown(f"""
            <div class="info-card">
                <div class="info-label">{label}</div>
                <div class="info-value">{value}</div>
                <div class="info-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">🌱 Leaf Analysis</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(
        [1.1, 0.9],
        gap="large"
    )

    image = None
    uploaded_file = None

    with left:

        st.markdown("""
        <div class="panel">
            <h3>Upload Leaf Image</h3>
            <p class="muted">
                Upload a clear photograph of a plant leaf.
                Supported formats: JPG, JPEG, PNG and WEBP.
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose a leaf image",
            type=["jpg", "jpeg", "png", "webp"],
            key=f"leaf_{st.session_state.upload_version}"
        )

        if uploaded_file is not None:

            try:

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                st.image(
                    image,
                    caption="Uploaded Leaf",
                    use_container_width=True
                )

                st.caption(
                    f"Image dimensions: {image.width} × {image.height}"
                )

            except Exception:

                st.error("Invalid image file.")

    with right:

        st.markdown("""
        <div class="panel">

            <h3>Analysis Workflow</h3>

            <p class="muted">
                01 · Image preprocessing
            </p>

            <p class="muted">
                02 · Mean RGB feature extraction
            </p>

            <p class="muted">
                03 · Spike-based neural processing
            </p>

            <p class="muted">
                04 · LIF neuron simulation
            </p>

            <p class="muted">
                05 · Output spike-rate comparison
            </p>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="panel">

            <h3>Model Summary</h3>

            <p class="muted">
                Input features: 3
            </p>

            <p class="muted">
                Hidden neurons: 8
            </p>

            <p class="muted">
                Output neurons: 2
            </p>

            <p class="muted">
                Simulation steps: 25
            </p>

        </div>
        """, unsafe_allow_html=True)

    st.write("")

    button1, button2 = st.columns([2, 1])

    with button1:

        analyze = st.button(
            "🔍 Analyze Plant Health",
            type="primary",
            use_container_width=True,
            disabled=image is None
        )

    with button2:

        st.button(
            "↻ Reset Analysis",
            use_container_width=True,
            on_click=reset_analysis
        )

    # -----------------------------------------------------
    # RUN PREDICTION
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
                "Analyzing leaf using Spiking Neural Network..."
            ):

                with torch.no_grad():

                    outputs = torch.stack([
                        model(x)[0]
                        for _ in range(N_FORWARD_PASSES)
                    ]).mean(0)

            healthy_rate = float(outputs[0].item())

            diseased_rate = float(outputs[1].item())

            prediction = int(outputs.argmax().item())

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

            st.error(f"Prediction failed: {error}")

    # -----------------------------------------------------
    # DISPLAY PREDICTION
    # -----------------------------------------------------

    result = st.session_state.prediction

    if result is not None:

        st.divider()

        st.markdown(
            '<div class="section-title">📋 Diagnostic Results</div>',
            unsafe_allow_html=True
        )

        if result["prediction"] == 0:

            st.markdown("""
            <div class="result-card healthy">

                <div class="result-label">
                    CLASSIFICATION RESULT
                </div>

                <div class="result-value">
                    🟢 HEALTHY
                </div>

                <div class="result-description">
                    The Healthy output neuron produced
                    the higher mean spike rate.
                </div>

            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="result-card diseased">

                <div class="result-label">
                    CLASSIFICATION RESULT
                </div>

                <div class="result-value">
                    🟠 DISEASED
                </div>

                <div class="result-description">
                    The Diseased output neuron produced
                    the higher mean spike rate.
                </div>

            </div>
            """, unsafe_allow_html=True)

        # Output metrics

        r1, r2 = st.columns(2)

        r1.metric(
            "Healthy Spike Rate",
            f'{result["healthy_rate"]:.4f}'
        )

        r2.metric(
            "Diseased Spike Rate",
            f'{result["diseased_rate"]:.4f}'
        )

        # Chart

        st.markdown(
            '<div class="section-title">📊 Output Spike Activity</div>',
            unsafe_allow_html=True
        )

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
            "forward passes. These are not calibrated probabilities."
        )

        # Report data

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

        st.markdown(
            '<div class="section-title">📥 Export Results</div>',
            unsafe_allow_html=True
        )

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

        with st.expander("🧠 View SNN Processing Details"):

            st.write("Architecture: 3 → 8 → 2")

            st.write("Neuron: Leaky Integrate-and-Fire")

            st.write(f"Time steps: {TIME_STEPS}")

            st.write(f"Beta: {BETA}")

            st.write(
                f"Forward passes: {N_FORWARD_PASSES}"
            )

            st.write("Input: Mean RGB values")

        with st.expander("⚠️ Research Limitations"):

            st.warning(
                "This prototype performs binary Healthy/Diseased "
                "classification. It does not identify exact disease types."
            )

            st.write(
                "Mean RGB feature extraction removes spatial lesion "
                "information from the leaf image."
            )

            st.write(
                f"Best recorded validation accuracy: "
                f"{VALIDATION_ACCURACY:.2f}%."
            )

            st.write(
                "This is a research prototype, not a validated "
                "agricultural diagnostic system."
            )


# =========================================================
# PAGE 2: SNN ARCHITECTURE
# =========================================================

elif page == "SNN Architecture":

    st.markdown(
        '<div class="section-title">🧠 SNN Architecture</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="panel">

        <h3>3 → 8 → 2 Neural Network</h3>

        <p class="muted">
            The network receives three mean RGB features,
            processes them through eight hidden LIF neurons,
            and produces two output spike rates.
        </p>

    </div>
    """, unsafe_allow_html=True)

    a, b, c = st.columns(3)

    for col, title, value, desc in [

        (a, "Input Layer", "3 Neurons", "Mean RGB features"),

        (b, "Hidden Layer", "8 Neurons", "LIF neurons"),

        (c, "Output Layer", "2 Neurons", "Healthy / Diseased")

    ]:

        with col:

            st.markdown(f"""
            <div class="info-card">

                <div class="info-label">{title}</div>

                <div class="info-value">{value}</div>

                <div class="info-sub">{desc}</div>

            </div>
            """, unsafe_allow_html=True)

    st.write("")

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
# PAGE 3: MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="section-title">📈 Model Performance</div>',
        unsafe_allow_html=True
    )

    p1, p2, p3 = st.columns(3)

    p1.metric(
        "Best Validation Accuracy",
        f"{VALIDATION_ACCURACY:.2f}%"
    )

    p2.metric(
        "Output Classes",
        "2"
    )

    p3.metric(
        "Input Resolution",
        "128 × 128"
    )

    st.markdown("""
    <div class="panel">

        <h3>Training Summary</h3>

        <p class="muted">
            The model was trained using the Plant Disease Dataset
            in Google Colab with CUDA acceleration.
        </p>

        <p class="muted">
            Classification task: Healthy vs Diseased.
        </p>

        <p class="muted">
            Best recorded validation accuracy: 70.87%.
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.warning(
        "The validation accuracy is a training result, not a "
        "prediction confidence score. Generalization to field images "
        "has not been established."
    )


# =========================================================
# PAGE 4: ABOUT PROJECT
# =========================================================

elif page == "About Project":

    st.markdown(
        '<div class="section-title">ℹ️ About the Project</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="panel">

        <h3>FPGA-Based Neuromorphic Plant Disease Detection</h3>

        <p class="muted">
            This research prototype explores the use of
            Spiking Neural Networks for plant health classification.
        </p>

        <p class="muted">
            The current software implementation uses a
            3-8-2 feed-forward SNN with Leaky Integrate-and-Fire neurons.
        </p>

        <p class="muted">
            The trained model is currently demonstrated through
            a Streamlit-based software interface.
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.subheader("Current Project Components")

    st.success("Dataset preparation — Completed")

    st.success("SNN model training — Completed")

    st.success("Model checkpoint export — Completed")

    st.success("Streamlit frontend — Running")

    st.info("FPGA hardware implementation — Pending")


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<div class="footer">

    <strong>🌿 AgroSNN</strong><br>

    FPGA-Based Neuromorphic Plant Disease Detection<br>

    Spiking Neural Network · Leaky Integrate-and-Fire<br>

    Research Prototype

</div>
""", unsafe_allow_html=True)
