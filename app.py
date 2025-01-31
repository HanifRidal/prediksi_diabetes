# import joblib
import numpy as np
# import scipy
# import threadpoolctl
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask, render_template, request, redirect
import pickle

app=Flask(__name__, template_folder='template')

@app.route('/', methods=['POST', 'GET'])

def index():
    if request.method == 'POST':
        # with open('knn_pickle.pkl', 'rb') as r:
        with open('knn_diabetes', 'rb') as r:
            model = pickle.load(r)

        melahirkan = float(request.form['melahirkan'])
        glukosa = float(request.form['glukosa'])
        darah = float(request.form['darah'])
        kulit = float(request.form['kulit'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        riwayat = float(request.form['riwayat'])
        umur = float(request.form['umur'])
        
        datas = np.array((melahirkan, glukosa, darah, kulit, insulin, bmi, riwayat, umur))
        datas = np.reshape(datas, (1, -1))

        isDiabetes = model.predict(datas)

        return render_template('hasil.html', finalData=isDiabetes)
    
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
