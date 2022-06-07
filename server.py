import asyncio
import json
import websockets


from gerenciador_logs import GerenciadorLogs
from gerenciador_arquivos import GerenciadorArquivos
from multilayer_perceptron import MultilayerPerceptron

logger = GerenciadorLogs()
gerenciador = GerenciadorArquivos("dados/caracteres-limpo.csv")
base_caracteres = gerenciador.le_csv(df_caracteres=True)
X_caracteres, y_caracteres = gerenciador.extrai_X_y(base_caracteres, True)

modelo = MultilayerPerceptron(len(X_caracteres[0]), 15, 7, tx_aprendizagem=0.1, gerenciador_logs=logger)

modelo.treina(X_caracteres, y_caracteres, epocas=5_000)

async def handler(websocket, path):
    data = await websocket.recv()
    request = json.loads(data)
    try:
        resultado = modelo.prediz(request["values"])
        await websocket.send(json.dumps(resultado))
    except Exception as error:
        print(error)
        await websocket.send(json.dumps({ "error": "Houve um erro ao fazer sua predição"}))


server = websockets.serve(handler, "localhost", 3333)
asyncio.get_event_loop().run_until_complete(server)
asyncio.get_event_loop().run_forever()