# encoding: utf-8
import sys
sys.path.append("../..")
from src.model_trainer.LazyLoader import LazyLoader
from src.model_trainer import rf_viewer
from src import config
from src.util import singleton


@singleton
class RfLoader(LazyLoader):
    def __init__(self):
        LazyLoader.__init__(self)
        for freq in range(1, 6):
            self._map_name_to_handler["nltk_unigram_withrf_t%d" % freq] = lambda: rf_viewer.Rf_Viewer(None, config.RF_DATA_NLTK_UNIGRAM_TU_PATH % freq)
            self._map_name_to_handler["nltk_bigram_withrf_t%d" % freq] = lambda: rf_viewer.Rf_Viewer(None, config.RF_DATA_NLTK_BIGRAM_TU_PATH % freq)
            self._map_name_to_handler["nltk_trigram_withrf_t%d" % freq] = lambda: rf_viewer.Rf_Viewer(None, config.RF_DATA_NLTK_TRIGRAM_TU_PATH % freq)
            self._map_name_to_handler["hashtag_withrf_t%d" % freq] = lambda: rf_viewer.Rf_Viewer(None, config.RF_DATA_HASHTAG_TU_PATH % freq)
