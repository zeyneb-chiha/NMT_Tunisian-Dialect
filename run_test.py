from model import nmt_dt2ar

if __name__ == "__main__":

    save_dir = 'model/model_save'
    nmt = nmt_dt2ar()
    nmt.load_model(save_dir=save_dir)
    nmt.translate_api_response(u'kamel')
