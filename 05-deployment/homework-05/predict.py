import pickle

model_file = 'model1.bin'
dv_file = 'dv.bin'

with open(model_file, 'rb') as f_in_model:
    model = pickle.load(f_in_model)

with open(dv_file, 'rb') as f_in_dv:
    dv = pickle.load(f_in_dv)


client = {'job': 'retired', 'duration': 445, 'poutcome': 'success'}

X = dv.transform([client])
y_pred = model.predict_proba(X)[0, 1]

print('client', client)
print('get credit probability', y_pred)

