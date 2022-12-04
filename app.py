from flask import Flask, render_template
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    data = np.array([0.1, 0.2, 0.3, 0.4]).reshape(1, -1)
    print(data)
    return render_template('index.html')




if __name__=='__main__':
    app.run(host='0.0.0.0', port=5432, debug=True)




