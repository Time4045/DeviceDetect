import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

df_columns = pd.read_csv('columns.csv').columns

#загрузка фрейма
data_list = []
with open('input.json') as file:
    for line in file:
        try:
            # Разбираем каждую строку как отдельный JSON-объект
            obj = json.loads(line.strip())
            data_list.append(obj)
        except json.JSONDecodeError as e:
            print(f"Ошибка при декодировании строки: {e}")

print(f'Count of JSON objects: {len(data_list)}')

keylist = [el.split('.') for el in df_columns]

def get_data(key, data):
    if len(key) == 1:
        key = key[0]
        try:
            return data[key]
        except Exception as e:
            return np.nan
    elif len(key) == 2:
        key1, key2, = key
        try:
            return data[key1][key2]
        except Exception as e:
            return np.nan
    elif len(key) == 3:
        key1, key2, key3 = key
        try:
            return data[key1][key2][key3]
        except Exception as e:
            return np.nan

data_frame = pd.DataFrame(columns=df_columns)
for data in data_list:
    obj = []
    for key in keylist:
        obj.append(get_data(key=key,data=data))
    prom_df = pd.DataFrame([obj], columns=df_columns)
    data_frame = pd.concat([data_frame, prom_df], ignore_index=True)


# Разделение akamai_fingerprint на составляющие
data_frame['akamai_fingerprint.settings'] = data_frame['http2.akamai_fingerprint'].apply(lambda x: x.split('|')[0])
data_frame['akamai_fingerprint.window_update'] = data_frame['http2.akamai_fingerprint'].apply(lambda x: x.split('|')[1])
data_frame['akamai_fingerprint.priority_weights'] = data_frame['http2.akamai_fingerprint'].apply(lambda x: x.split('|')[2])
data_frame['akamai_fingerprint.flags'] = data_frame['http2.akamai_fingerprint'].apply(lambda x: x.split('|')[3])

# Удаление исходного столбца
data_frame.drop(columns=['http2.akamai_fingerprint'], inplace=True)

#разделение ja3 на составляющие
data_frame['tls_version'] = data_frame['tls.ja3'].apply(lambda x: x.split(',')[0])
data_frame['ciphers'] = data_frame['tls.ja3'].apply(lambda x: x.split(',')[1])
data_frame['expansions'] = data_frame['tls.ja3'].apply(lambda x: x.split(',')[2])
data_frame['supp_gr'] = data_frame['tls.ja3'].apply(lambda x: x.split(',')[3])
data_frame['signature_algorithms'] = data_frame['tls.ja3'].apply(lambda x: x.split(',')[4])

#признаки, которые должен содержать df
cols = ['ip', 'http_version', 'method',
       'tls.tls_version_negotiated', 'tls.ja4', 'http2.akamai_fingerprint_hash', 'tcpip.ip.ttl',
       'tcpip.ip.ip_version', 'tcpip.tcp.mss', 'tcpip.tcp.window',
       'blacklist_info.blacklist_info.TotalBlacklists',
       'blacklist_info.blacklist_info.CountFoundIn',
       'akamai_fingerprint.settings', 'akamai_fingerprint.window_update',
       'akamai_fingerprint.priority_weights', 'akamai_fingerprint.flags',
       'tls_version', 'ciphers', 'expansions', 'supp_gr',
       'signature_algorithms']
data_frame = data_frame[cols]
#удаление пропущенных значений
data_frame.dropna(inplace=True)

#смена типа данных
columns_for_switch_type = ['tls.tls_version_negotiated','tcpip.ip.ttl','tcpip.ip.ip_version','tcpip.tcp.mss',
                          'tcpip.tcp.window','blacklist_info.blacklist_info.TotalBlacklists','blacklist_info.blacklist_info.CountFoundIn',
                          'akamai_fingerprint.window_update','tls_version']
for col in columns_for_switch_type:
    data_frame[col] = data_frame[col].astype('int')

# Android: 0
# Linux: 1
# Mac OS: 2
# Windows: 3
# iOS: 4

#сохранение ip
ips = data_frame.ip

#кодирование категориальных признаков
for col in data_frame.select_dtypes('object').columns:
    data_frame[col] = LabelEncoder().fit_transform(data_frame[col])

#загрузка модели
with open('random_forest_model_version_2.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

#выдает результат предсказания (если type = console то выводит в консоль,
#если type = file то сохраняет результат в json
def get_result(df, model = loaded_model, type_ = 'console'):
    os = {0: 'Android', 1: 'Linux', 2: 'Mac OS', 3: 'Windows', 4: 'iOS'}
    if type_ == 'console':
        predict_scores = model.predict_proba(df)
        res = []
        for i in range(len(predict_scores)):
            res_d = {
                'ip': ips[i],
                'predicted_os': os[predict_scores[i].argmax()],
                'confidence': predict_scores[i].max().item()
            }
            res.append(res_d)
        for el in res:
            print(el)
    elif type_ == 'file':
        predict_scores = model.predict_proba(df)
        res = []
        for i in range(len(predict_scores)):
            res_d = {
                'ip': ips[i],
                'predicted_os': os[predict_scores[i].argmax()],
                'confidence': predict_scores[i].max().item()
            }
            res.append(res_d)
        with open('output.json', 'w', encoding='utf-8') as file:
            json.dump(res, file, ensure_ascii=False, indent=4)

get_result(data_frame)
