# Neural Style Transfer 🎨

## About the Project

Neural Style Transfer is a deep learning project that combines the content of one image with the artistic style of another image.

For example, you can take an ordinary photograph and apply the painting style of a famous artwork to it while keeping the original content recognizable.

I built this project to explore how deep learning can be used creatively and to gain hands-on experience with PyTorch and VGG-based feature extraction.

## How It Works

The project uses a deep learning approach to understand two important things:

- **Content Image:** The image whose main objects and structure we want to preserve.
- **Style Image:** The image whose artistic patterns, textures, and colors we want to apply.

The model extracts content and style features and combines them to generate a new stylized image.

## Technologies Used

- Python
- PyTorch
- Deep Learning
- VGG (Feature Extraction)
- Flask
- HTML and CSS

## Key Features

- Upload a content image.
- Upload a reference style image.
- Apply artistic styles to the content image.
- Generate a new stylized image.
- View the generated result.

## Project Structure

```text
Neural-Style-Transfer/
│
├── app.py
├── utils/
├── templates/
├── static/
├── experiment/
├── vgg_normalised.pth
├── decoder_final.pth
└── README.md
```

## Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/Vrushali-726/Neural-Style-Transfer-Project.git
```

**2. Navigate to the project folder**

```bash
cd Neural-Style-Transfer-Project
```

**3. Install the required dependencies**

Install the packages listed in your project's dependency file.

**4. Run the application**

```bash
python app.py
```

Open the local URL displayed in your terminal to access the application.

## What I Learned

While building this project, I learned about:

- How neural networks extract image features.
- VGG-based feature extraction.
- The difference between content and style representations.
- Working with PyTorch and deep learning models.
- Integrating a deep learning model with a Flask application.

## Future Improvements

- Support more image formats.
- Improve image generation speed.
- Add more artistic style options.
- Improve the user interface.

## Author

**Vrushali Solanke**

Computer Engineering Student | AI & Machine Learning Enthusiast

[GitHub Profile](https://github.com/Vrushali-726)

---

*Built with curiosity and a passion for exploring Artificial Intelligence.*
