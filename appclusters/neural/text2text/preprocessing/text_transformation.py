import json
import logging

from appclusters.util.strings import remove_html
from appclusters.util.text_processing import tokenize_text
from appclusters.util.tfidf_model_creator import TfIdfModelCreator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

# TODO: update preprocessing for new samples database

def meta_data_description_tokenize(file_path):
    fp = open(file_path, "r")
    meta = MetaInfo(json.load(fp))
    text = remove_html(meta.description_text)
    tokens = tokenize_text(text)
    return tokens


def get_relevant_files():
    samples_props = SamplePropertiesLoader()

    def relevant_samples(sample):
        criterion_lang = sample[Properties.LANGUAGE.value] == 'en'
        return criterion_lang

    packages = samples_props.filter_by(relevant_samples)
    files = ["%s.json" % f for f in packages]

    return files, packages


def run():
    files, packages = get_relevant_files()

    model_creator = TfIdfModelCreator(model_file=config,
                                      data_file=config.Pre.tf_idf_data,
                                      files_folder=config.SourceData.metafiles_folder,
                                      files=files,
                                      file_ids=packages,
                                      tokenize_function=meta_data_description_tokenize)
    model_creator.create_model()
    model_creator.transform_data()
    model_creator.load()
    model_creator.info()


if __name__ == "__main__":
    run()
