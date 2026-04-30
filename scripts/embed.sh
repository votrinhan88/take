#!/bin/bash
# Meta: --base_config STR --run INT --n_runs INT
# Args: --dataset {agnews,imdb,mnlim,qqp,sst2,qnli}
#       --encoder {e5,glove,jina_nano,jina_small,minilm,tfidf}
# Opts: --embed_dim INT --batch_size INT --path_output STR

source /vast/s223032975/textdd/.env
cd "$PROJECT"
"$MAIN_SH" --env_path="$PROJECT/.venv" --python_path="$PROJECT/expts/embed.py" -- "$@"
