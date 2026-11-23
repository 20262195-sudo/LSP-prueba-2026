"""Entrenamiento de poses estáticas (RF) y secuencias (LSTM opcional)."""
from __future__ import annotations
import json
from collections import Counter
import numpy as np
from motor import (DIM,MODELS,STATIC_MODEL,DYNAMIC_MODEL,DYNAMIC_LABELS,SEQUENCE_LEN,
                   load_static,load_sequences,prepare_dirs)


def _report(title,labels,actual,pred,kind,notes):
    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,balanced_accuracy_score
    title_lines=[title, "", f"Cantidad de muestras de evaluación: {len(actual)}",
                 f"Exactitud en división aleatoria: {accuracy_score(actual,pred):.3f}",
                 f"Exactitud balanceada: {balanced_accuracy_score(actual,pred):.3f}", "",
                 "REPORTE POR CLASE",classification_report(actual,pred,labels=labels,zero_division=0),
                 "MATRIZ DE CONFUSIÓN (filas = real, columnas = predicción)",
                 "Orden: "+', '.join(labels),str(confusion_matrix(actual,pred,labels=labels)),
                 "",notes]
    content='\n'.join(title_lines)
    dest=MODELS/f'reporte_{kind}.txt'
    dest.write_text(content,encoding='utf-8')
    return content


def train_static():
    from joblib import dump
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    prepare_dirs()
    rows=load_static()
    distribution=Counter(row['label'] for row in rows)
    if len(distribution)<2 or min(distribution.values())<20:
        raise ValueError("Necesitas al menos 2 etiquetas diferentes con 20 muestras cada una")
    X=np.stack([np.asarray(r['feature'],dtype=np.float32) for r in rows])
    y=np.asarray([r['label'] for r in rows])
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=.25,stratify=y)
    model=RandomForestClassifier(n_estimators=200,class_weight='balanced',n_jobs=-1,random_state=42)
    model.fit(X_train,y_train)
    guesses=model.predict(X_test)
    dump({'model':model,'dim':DIM,'kind':'static'},STATIC_MODEL)
    return _report("PROTOTIPO LSP: RECONOCIMIENTO ESTÁTICO / RANDOM FOREST",
                   sorted(distribution),y_test,guesses,'estatico',
                   "IMPORTANTE: evaluación interna; una separación aleatoria entre muestras de una misma persona "
                   "puede sobreestimar la precisión real. Valida con personas y sesiones nuevas. "
                   "Las etiquetas solo representan LSP si están verificadas externamente.")


def train_dynamic():
    # Instalar TensorFlow solo si se va a usar el modelo de secuencias.
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("Para entrenar la LSTM necesitas ejecutar INSTALAR_LSTM.bat una vez") from exc
    from sklearn.model_selection import train_test_split
    prepare_dirs()
    rows=load_sequences()
    distribution=Counter(label for label,_,_ in rows)
    if len(distribution)<2 or min(distribution.values())<12:
        raise ValueError("Necesitas al menos 2 clases con 12 secuencias cada una; se recomiendan 30 o más")
    labels=sorted(distribution)
    indices={k:i for i,k in enumerate(labels)}
    X=np.stack([x for _,x,_ in rows]).astype(np.float32)
    yi=np.asarray([indices[label] for label,_,_ in rows],dtype=np.int64)
    tr,te=train_test_split(np.arange(len(yi)),test_size=.25,random_state=42,stratify=yi)
    tf.keras.utils.set_random_seed(42)
    model=tf.keras.Sequential([
        tf.keras.layers.Input((SEQUENCE_LEN,DIM)),
        tf.keras.layers.Masking(mask_value=0.),
        tf.keras.layers.LSTM(64,return_sequences=True),
        tf.keras.layers.Dropout(.25),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(32,activation='relu'),
        tf.keras.layers.Dense(len(labels),activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
                  loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(X[tr],yi[tr],validation_split=.2,epochs=35,batch_size=8,verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=6,restore_best_weights=True)])
    prediction=np.argmax(model.predict(X[te],verbose=0),axis=1)
    model.save(DYNAMIC_MODEL)
    DYNAMIC_LABELS.write_text(json.dumps({'labels':labels,'length':SEQUENCE_LEN,'dim':DIM},ensure_ascii=False,indent=2),encoding='utf-8')
    real_labels=[labels[int(v)] for v in yi[te]]
    predict_labels=[labels[int(v)] for v in prediction]
    return _report("PROTOTIPO LSP: SECUENCIAS / LSTM",
                   labels,real_labels,predict_labels,'lstm',
                   "IMPORTANTE: LSTM entrenada con los datos grabados por el equipo, NO preentrenada en LSP. "
                   "Una división aleatoria de secuencias del mismo firmante puede exagerar la precisión. "
                   "Evaluar con nuevos firmantes, buena iluminación y señas verificadas.")


def load_static_model():
    from joblib import load
    if not STATIC_MODEL.exists():
        raise FileNotFoundError("Primero entrena el modelo estático")
    pack=load(STATIC_MODEL)  # Cargar solo el modelo generado por el equipo; joblib ejecuta pickle.
    if pack.get('dim')!=DIM:
        raise ValueError("Modelo incompatible con versión de rasgos del programa")
    return pack['model']


def load_dynamic_model():
    import tensorflow as tf
    if not DYNAMIC_MODEL.exists() or not DYNAMIC_LABELS.exists():
        raise FileNotFoundError("Primero entrena la LSTM")
    config=json.loads(DYNAMIC_LABELS.read_text(encoding='utf-8'))
    if config.get('dim')!=DIM or config.get('length')!=SEQUENCE_LEN:
        raise ValueError("LSTM incompatible con longitud de secuencias")
    return tf.keras.models.load_model(DYNAMIC_MODEL),config['labels']
