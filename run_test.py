from model import nmt_ar2en

if __name__ == "__main__":

    save_dir = 'model/model_save'
    nmt = nmt_ar2en()
    nmt.load_model(save_dir=save_dir)
    nmt.translate_api_response(u'kamel')
