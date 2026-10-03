from flask import Flask, render_template, request
import checking

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html', prediction=None, error=None)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = [
            float(request.form.get('pregnancies', 0)),
            float(request.form.get('glucose', 0)),
            float(request.form.get('bp', 0)),
            float(request.form.get('skin', 0)),
            float(request.form.get('insulin', 0)),
            float(request.form.get('bmi', 0)),
            float(request.form.get('dpf', 0)),
            float(request.form.get('age', 0)),
        ]

        result = checking.predict_diabetes(values)
        output = 'Diabetic' if result == 1 else 'Not Diabetic'

        return render_template('index.html', prediction=output, error=None)
    except ValueError:
        return render_template('index.html', prediction=None, error='Please enter valid numeric values.')
    except Exception as exc:
        return render_template('index.html', prediction=None, error=f'Prediction failed: {exc}')


if __name__ == '__main__':
    app.run(debug=True)
