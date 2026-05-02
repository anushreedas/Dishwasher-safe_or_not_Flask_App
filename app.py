from flask import Flask, render_template, request
from classify import preprocess_img, predict_result
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/"


@app.route("/")
def main():
    return render_template("index.html")


@app.route('/prediction', methods=['POST'])
def predict_image_file():
    try:
        if request.method == 'POST':
            f = request.files['file']
            filename = secure_filename(f.filename)
            img_path = app.config['UPLOAD_FOLDER'] + filename
            f.save(img_path)

            img = preprocess_img(img_path)
            pred = predict_result(img)

            return render_template(
                "index.html",
                predictions=str(pred),
                img_filename=filename
            )

    except Exception as e:
        print(f"ERROR: {e}")          # <-- add this line temporarily
        import traceback
        traceback.print_exc()          # <-- prints the full stack trace
        return render_template(
            "index.html",
            err=f"File cannot be processed. Detail: {str(e)}"
        )

if __name__ == "__main__":
    app.run(port=9000, debug=True)