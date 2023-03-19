from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import numpy as np
app = Flask(__name__)
CORS(app)



model=pickle.load(open('model2.pkl','rb'))

@app.route('/', methods=['POST'])
def index():
    tests={'(vertigo) Paroymsal  Positional Vertigo':'Dix-Hallpike maneuver',
    'AIDS':'Nucleic Acid Test (NAT)',
    'Acne':'None required',
    'Alcoholic hepatitis':'Liver function tests, Blood tests',
    'Allergy':'RAST (Radioallergosorbent test)',
    'Arthritis':'Joint scans',
    'Bronchial Asthma':'Spirometry',
    'Cervical spondylosis':'Neck X-ray, MRI, CT myel',
    'Chicken pox':'Whole infected cell (wc) ELISA',
    'Chronic cholestasis':'Abdominal Ultrasonography',
    'Common Cold':'None required',
    'Dengue':'Nucleic acid amplification tests (NAATs)',
    'Diabetes':'Blood sugar test',
    'Dimorphic hemmorhoids(piles)':'Anoscopy',
    'Drug Reaction':'Skin tests',
    'Fungal infection':'Fungal culture test',
    'GERD':'Upper gastrointestinal (GI) endoscopy',
    'Gastroenteritis':'rapid stool test',
    'Heart attack':'Electrocardiogram (ECG or EKG)',
    'Hepatitis B':'Blood tests, Liver ultrasound, Liver biopsy',
    'Hepatitis C':'HCV antibody test',
    'Hepatitis D':'anti-HDV immunoglobulin G (IgG) and immunoglobulin M (IgM) test',
    'Hepatitis E':'specific anti-HEV immunoglobulin M (IgM) test',
    'Hypertension ':'Muscle biopsy (contracture test)',
    'Hyperthyroidism':'Thyroid hormone blood test, T4and T3',
    'Hypoglycemia': 'Glucagon test',
    'Hypothyroidism':'Thyroid hormone blood test, T4and T3',
    'Impetigo':'Culture of the exudate or pus from an impetigo lesion',
    'Jaundice':'Bilirubin blood test',
    'Malaria':'Smear microscopy, RDT, PCR',
    'Migraine':'Reocrds of frequent headaches',
    'Osteoarthristis':'Xray, Reflex test, General health test',
    'Paralysis (brain hemorrhage)':'CT Scan',
    'Peptic ulcer diseae':'Esophagogastroduodenoscopy (EGD)',
    'Pneumonia':'Chest xray, CBC test',
    'Psoriasis':'Skin Biopsy',
    'Tuberculosis':'Mantoux tuberculin skin test (TST), TB blood test',
    'Typhoid':'Body fluid or tissue culture test',
    'Urinary tract infection':'Urinalysis',
    'Varicose veins':'venous Doppler ultrasound',
    'hepatitis A':'HAV-specific immunoglobulin G (IgM) antibodies blood test'
    }
    if request.method == 'POST':
        l=[]
        data = request.get_json()
        #print(data)
        for x in data.values():
            l.append(x)
        #print(l)
        predict_name=model.predict([l])
        prediction=model.predict_proba([l])   
        j=np.amax([prediction[0]])
        j=j*100
        print(predict_name)
        return jsonify({"ans":predict_name[0],"ans_prob": j ,"test": tests[predict_name[0]]})

if __name__ == '__main__':
   app.run()
   
   