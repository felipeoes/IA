import asyncio
import json
import websockets
import os

from gerenciador_logs import GerenciadorLogs
from gerenciador_arquivos import GerenciadorArquivos
from multilayer_perceptron import MultilayerPerceptron

logger = GerenciadorLogs()
gerenciador = GerenciadorArquivos("dados/caracteres-limpo.csv")
base_caracteres = gerenciador.le_csv(df_caracteres=True)
X_caracteres, y_caracteres = gerenciador.extrai_X_y(base_caracteres, True)


# Checa se existe um json do modelo salvo para não precisar treinar o modelo toda vez que o servidor for iniciado

def json_keys_to_int(x):
    if isinstance(x, dict):
        try:
            return {int(k): v for k, v in x.items()}
        except:
            pass
    return x


if os.path.exists("modelo_treinado.json"):
    with open("modelo_treinado.json", "r") as arq_json:
        json_lido = json.loads(arq_json.read(), object_hook=json_keys_to_int)
        modelo = MultilayerPerceptron.from_json(json_lido)

else:
    modelo = MultilayerPerceptron(len(
        X_caracteres[0]), 15, 7, tx_aprendizagem=0.1, gerenciador_logs=logger)
    modelo.treina(X_caracteres, y_caracteres, epocas=15000)
    log_mlp = logger.log_html()

    with open("log_mlp.html", "w") as arq:
        arq.write(log_mlp)

    with open("modelo_treinado.json", "w") as arq_json:
        arq_json.write(json.dumps(modelo.to_json()))


async def handler(websocket, path):
    data = await websocket.recv()
    request = json.loads(data)
    try:
        resultado = modelo.prediz(request["values"])
        await websocket.send(json.dumps(resultado))
    except Exception as error:
        print(error)
        await websocket.send(json.dumps({"error": "Houve um erro ao fazer sua predição"}))


server = websockets.serve(handler, "localhost", 3333)
asyncio.get_event_loop().run_until_complete(server)
asyncio.get_event_loop().run_forever()
