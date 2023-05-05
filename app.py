import os
from potassium import Potassium, Request, Response

from detection import checkBox_Ops, loadModel
import helper as hp
import env_config


app = Potassium("my_app")


if env_config.RETRIEVE_AWS_CONFIGS:
    hp.fill_common_configs_from_aws()


# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    print('-----loading model--------')
    model = loadModel("cb", num_class=3)
   
    context = {
        "model": model
    }
    print('-----model loaded--------')

    return context

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    file_path = request.json.get('page_path', '')
    file_id = request.json.get('doc_id', '')
    data = request.json.get('checkbox_data')
    print('-----predicting--------')
    
    cb_predictor = context.get("model")
   
    
    file_name = '{}_{}'.format(file_id, os.path.split(file_path)[1])
    if not os.path.exists(env_config.TEMP_FILES_PATH):
        os.mkdir(env_config.TEMP_FILES_PATH)
    
    img = hp.download_page_s3(file_path, file_name, file_id,
                              env_config.TEMP_FILES_PATH)
                              
    print('-----downloading image from S3--------')
    return_data = []

    for i in data:
        if i['obj_type'] == 'CheckBox':
            vx1, vy1, vx2, vy2 = [int(v) for
                                    v in i['value_roi'].split(",")]
            valueImg = img[vy1: vy2, vx1: vx2]
            updated_data = checkBox_Ops(
                i, valueImg, cb_predictor)
            return_data.append(updated_data)
        elif i['obj_type'] in ('Table', 'Table_Key_CheckBox'):
            for cidx, cell in enumerate(i['value_text']):
                if cell['cell_type'] == "CheckBox":
                    vx1, vy1, vx2, vy2 = [int(v) for v in
                                            cell['value_roi'].split(",")]
                    valueImg = img[vy1: vy2, vx1: vx2]
                    updated_data = checkBox_Ops(cell, valueImg,
                                                cb_predictor)
                    update_cell = {"cell_roi": updated_data['value_roi'],
                                    "cell_type": "CheckBox",
                                    "cell_value": updated_data['value_text'],
                                    "element_ID": updated_data['element_ID']}
                    i['value_text'][cidx] = update_cell
            return_data.append(i)
        else:
            pass
    print('-----completed prediction--------')

    return Response(
        json = {"Response": return_data}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()