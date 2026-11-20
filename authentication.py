import streamlit as st
import secrets
import hashlib
import time
import re
import smtplib
from email.message import EmailMessage


OTP_EXPIRY = 300       # 5 minutes
RESEND_COOLDOWN = 30  # 30 seconds
MAX_ATTEMPTS = 5


# ---------------- SESSION STATE ----------------

def initialize_auth():
    defaults = {
        "authenticated": False,
        "auth_step": "login",
        "otp_hash": None,
        "otp_expiry": 0,
        "otp_sent_at": 0,
        "otp_attempts": 0,
        "user_email": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------- OTP GENERATION ----------------

def generate_otp():
    return f"{secrets.randbelow(900000) + 100000}"


def hash_otp(otp):
    return hashlib.sha256(otp.encode()).hexdigest()


# ---------------- SEND EMAIL ----------------

def send_email_otp(receiver, otp):

    sender = st.secrets["EMAIL_ADDRESS"]
    app_password = st.secrets["EMAIL_PASSWORD"]

    message = EmailMessage()
    message["Subject"] = "AgroSNN - Email Verification"
    message["From"] = sender
    message["To"] = receiver

    message.set_content(f"""
Welcome to AgroSNN!

Your verification code is:

{otp}

This OTP is valid for 5 minutes.

If you did not request this code, please ignore this email.
Do not share this code with anyone.
""")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as smtp:
        smtp.login(sender, app_password)
        smtp.send_message(message)


# ---------------- AUTHENTICATION UI ----------------

def authentication_ui():

    initialize_auth()

    if st.session_state.authenticated:
        return True

    st.markdown("""
    <style>
    .auth-title {
        text-align: center;
        font-size: 34px;
        font-weight: 700;
    }

    .auth-subtitle {
        text-align: center;
        color: #9ca3af;
        margin-bottom: 25px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ================= LOGIN SCREEN =================

    if st.session_state.auth_step == "login":

        st.markdown(
            "<h1 class='auth-title'>Welcome to AgroSNN</h1>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<p class='auth-subtitle'>Sign in using your email</p>",
            unsafe_allow_html=True
        )

        email = st.text_input(
            "Email address",
            placeholder="you@example.com"
        ).strip().lower()

        if st.button("Send OTP", use_container_width=True):

            if not re.fullmatch(
                r"[^@\s]+@[^@\s]+\.[^@\s]+",
                email
            ):
                st.error("Please enter a valid email address.")

            else:
                otp = generate_otp()

                try:
                    with st.spinner("Sending verification email..."):
                        send_email_otp(email, otp)

                    st.session_state.user_email = email
                    st.session_state.otp_hash = hash_otp(otp)
                    st.session_state.otp_expiry = time.time() + OTP_EXPIRY
                    st.session_state.otp_sent_at = time.time()
                    st.session_state.otp_attempts = 0
                    st.session_state.auth_step = "otp"

                    st.success("OTP sent to your email!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Email sending failed: {e}")

    # ================= OTP SCREEN =================

    elif st.session_state.auth_step == "otp":

        st.header("Verify Your Email")

        st.info(f"Verification code sent to: {st.session_state.user_email}")

        otp_input = st.text_input(
            "Enter 6-digit OTP",
            max_chars=6,
            type="password",
            placeholder="Enter OTP"
        )

        remaining_expiry = int(
            st.session_state.otp_expiry - time.time()
        )

        if remaining_expiry <= 0:
            st.error("OTP has expired. Please request a new one.")

        elif st.session_state.otp_attempts >= MAX_ATTEMPTS:
            st.error("Too many attempts. Please request a new OTP.")

        else:
            st.caption(
                f"OTP expires in {remaining_expiry} seconds."
            )

        if st.button("Verify & Continue", use_container_width=True):

            if time.time() > st.session_state.otp_expiry:
                st.error("OTP expired. Please request a new OTP.")

            elif st.session_state.otp_attempts >= MAX_ATTEMPTS:
                st.error("Too many attempts. Please request a new OTP.")

            elif not otp_input.isdigit() or len(otp_input) != 6:
                st.error("Enter a valid 6-digit OTP.")

            elif hash_otp(otp_input) == st.session_state.otp_hash:
                st.session_state.authenticated = True
                st.session_state.auth_step = "authenticated"
                st.session_state.otp_hash = None

                st.success("Email verified successfully!")
                st.rerun()

            else:
                st.session_state.otp_attempts += 1
                remaining_attempts = (
                    MAX_ATTEMPTS - st.session_state.otp_attempts
                )

                st.error(
                    f"Incorrect OTP. {remaining_attempts} "
                    "attempts remaining."
                )

        # -------- RESEND OTP --------

        elapsed = time.time() - st.session_state.otp_sent_at
        cooldown_remaining = max(
            0,
            RESEND_COOLDOWN - int(elapsed)
        )

        if cooldown_remaining > 0:
            st.caption(
                f"You can resend OTP in {cooldown_remaining} seconds."
            )

        else:
            if st.button("Resend OTP"):

                new_otp = generate_otp()

                try:
                    with st.spinner("Sending new OTP..."):
                        send_email_otp(
                            st.session_state.user_email,
                            new_otp
                        )

                    st.session_state.otp_hash = hash_otp(new_otp)
                    st.session_state.otp_expiry = time.time() + OTP_EXPIRY
                    st.session_state.otp_sent_at = time.time()
                    st.session_state.otp_attempts = 0

                    st.success("New OTP sent successfully!")
                    st.rerun()

                except Exception:
                    st.error("Failed to resend OTP. Please try again.")

        # -------- CHANGE EMAIL --------

        if st.button("Change Email"):

            st.session_state.auth_step = "login"
            st.session_state.otp_hash = None
            st.session_state.user_email = ""
            st.session_state.otp_attempts = 0

            st.rerun()

    return st.session_state.authenticated