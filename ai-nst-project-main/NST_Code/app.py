
import os
import torch
from flask import Flask, render_template, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# --------------------------------------------------
# Flask Configuration
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config["SECRET_KEY"] = "supersecretkey"
app.config["UPLOAD_FOLDER"] = os.path.join(
    BASE_DIR, "static", "uploads"
)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

Bootstrap(app)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


# --------------------------------------------------
# Upload Form
# --------------------------------------------------

class UploadForm(FlaskForm):
    content = FileField("Content Image")
    style = FileField("Style Image")
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField("Alpha", default=1.0)
    submit = SubmitField("Transfer Style")


# --------------------------------------------------
# Load Trained Models
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder_path = os.path.join(BASE_DIR, "vgg_normalised.pth")
decoder_path = os.path.join(
    BASE_DIR, "experiment", "final_exp", "decoder_final.pth"
)

if not os.path.isfile(encoder_path):
    raise FileNotFoundError(f"VGG weights not found: {encoder_path}")

if not os.path.isfile(decoder_path):
    raise FileNotFoundError(f"Decoder weights not found: {decoder_path}")

encoder = VGGEncoder(encoder_path).to(device)
decoder = Decoder().to(device)

decoder.load_state_dict(
    torch.load(decoder_path, map_location=device)
)

encoder.eval()
decoder.eval()

print(f"Models loaded successfully on {device}")


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def save_uploaded_file(file):
    if not file or not file.filename:
        return None

    if not allowed_file(file.filename):
        raise ValueError("Please upload a PNG, JPG, or JPEG image.")

    filename = secure_filename(file.filename)

    if not filename:
        raise ValueError("Invalid filename.")

    # Avoid overwriting files with the same name.
    name, extension = os.path.splitext(filename)
    filename = f"{name}_{os.urandom(4).hex()}{extension}"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    return filename


def style_transfer(content_image, style_image, alpha):
    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    content_tensor = (
        content_transform(content_image).unsqueeze(0).to(device)
    )
    style_tensor = (
        style_transform(style_image).unsqueeze(0).to(device)
    )

    with torch.no_grad():
        content_features = encoder(content_tensor, is_test=True)
        style_features = encoder(style_tensor, is_test=True)

        stylized_features = adaptive_instance_normalization(
            content_features, style_features
        )

        stylized_features = (
            alpha * stylized_features
            + (1 - alpha) * content_features
        )

        stylized_image = decoder(stylized_features)

    return stylized_image


def save_image(image, path):
    image = image.detach().cpu().clone().squeeze(0)
    image = image.clamp(0, 1)

    image = transforms.ToPILImage()(image)
    image.save(path)


# --------------------------------------------------
# Main Route
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    form = UploadForm()

    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():
        try:
            content_filename = save_uploaded_file(form.content.data)
            style_filename = save_uploaded_file(form.style.data)

            if not content_filename or not style_filename:
                raise ValueError(
                    "Please upload both a content image and a style image."
                )

            alpha = form.alpha.data

            if alpha is None or not 0.0 <= alpha <= 1.0:
                raise ValueError("Alpha must be between 0 and 1.")

            content_path = os.path.join(
                app.config["UPLOAD_FOLDER"], content_filename
            )
            style_path = os.path.join(
                app.config["UPLOAD_FOLDER"], style_filename
            )

            with Image.open(content_path) as img:
                content_image = img.convert("RGB")

            with Image.open(style_path) as img:
                style_image = img.convert("RGB")

            print("Starting style transfer...")

            output = style_transfer(
                content_image, style_image, alpha
            )

            result_filename = "stylized_" + content_filename
            result_path = os.path.join(
                app.config["UPLOAD_FOLDER"], result_filename
            )

            save_image(output, result_path)

            if not os.path.isfile(result_path):
                raise RuntimeError("Output image was not saved.")

            result_image = result_filename
            print(f"Style transfer completed: {result_path}")

        except Exception as e:
            app.logger.exception("Style transfer failed")
            error = str(e)

    elif form.is_submitted():
        print("FORM ERRORS:", form.errors)
        error = "Form validation failed. Please check the uploaded images and try again."

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )


# --------------------------------------------------
# Image Routes
# --------------------------------------------------

@app.route("/uploads/<filename>")
def send_image(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"], filename
    )


@app.route("/examples/<path:filename>")
def send_example(filename):
    examples_dir = os.path.join(BASE_DIR, "examples")
    return send_from_directory(examples_dir, filename)


# --------------------------------------------------
# Run Application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
