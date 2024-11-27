#! /bin/sh
#
# Copyright (C) 2021 
#
# Distributed under terms of the MIT license.
#
dataset_list=(steam_player IMDB_dir_form IMDB_dir_genre IMDB_wri_form IMDB_wri_genre twitter_friend)
khwl_type=fwl
list=(64)
layers=(3)
lr=0.001
weight_decay=1e-5
cuda=0
model_name=$1
runs=5
epochs=50
for dataset in ${dataset_list[*]}
do
    if [ $dataset = "RHG_3" ]; then
      for hid_MLP in ${list[*]}
      do
        for la in ${layers[*]}
        do
          echo =============
          echo ">>>>  Model:$model_name (default), Dataset: ${dataset} with mlp ${hid_MLP} with layers ${la}"
          echo "start at $(date +%R)"
          python main_graph.py \
              --model_name $model_name \
              --dataset $dataset \
              --All_num_layers $la \
              --MLP_num_layers 2 \
              --heads 1 \
              --MLP_hidden 128 \
              --KHWL_num_layers 2 \
              --KHWL_hidden $hid_MLP \
              --wd $weight_decay \
              --kwl_type $khwl_type \
              --epochs $epochs \
              --runs $runs \
              --cuda $cuda \
              --lr $lr
          echo "end at $(date +%R)"
          echo "Finished training on ${dataset} with mlp ${hid_MLP} with layers ${la}"
          echo =============
          done
      done
    elif [ "$dataset" = "RHG_10" ]; then
          for hid_MLP in ${list[*]}
          do
            for la in ${layers[*]}
            do
              echo =============
              echo ">>>>  Model:$model_name (default), Dataset: ${dataset} with mlp ${hid_MLP} with layers ${la}"
              echo "start at $(date +%R)"
              python main_graph.py \
                  --model_name $model_name \
                  --dataset $dataset \
                  --All_num_layers $la \
                  --MLP_num_layers 2 \
                  --heads 1 \
                  --MLP_hidden 128 \
                  --KHWL_num_layers 2 \
                  --KHWL_hidden $hid_MLP \
                  --wd $weight_decay \
                  --kwl_type $khwl_type \
                  --epochs $epochs \
                  --runs $runs \
                  --cuda $cuda \
                  --lr $lr
              echo "end at $(date +%R)"
              echo "Finished training on ${dataset} with mlp ${hid_MLP} with layers ${la}"
              echo =============
            done
          done
    elif [ "$dataset" = "RHG_table" ]; then
          for hid_MLP in ${list[*]}
          do
            for la in ${layers[*]}
            do
              echo =============
              echo ">>>>  Model:$model_name (default), Dataset: ${dataset} with mlp ${hid_MLP} with layers ${la}"
              echo "start at $(date +%R)"
              python main_graph.py \
                  --model_name $model_name \
                  --dataset $dataset \
                  --All_num_layers $la \
                  --MLP_num_layers 2 \
                  --heads 1 \
                  --MLP_hidden 128 \
                  --KHWL_num_layers 2 \
                  --KHWL_hidden $hid_MLP \
                  --wd $weight_decay \
                  --kwl_type $khwl_type \
                  --epochs $epochs \
                  --runs $runs \
                  --cuda $cuda \
                  --lr $lr
              echo "end at $(date +%R)"
              echo "Finished training on ${dataset} with mlp ${hid_MLP} with layers ${la}"
              echo =============
            done
          done
    elif [ "$dataset" = "RHG_pyramid" ]; then
        for hid_MLP in ${list[*]}
        do
          for la in ${layers[*]}
          do
            echo =============
            echo ">>>>  Model:$model_name (default), Dataset: ${dataset} with mlp ${hid_MLP} with layers ${la}"
            echo "start at $(date +%R)"
            python main_graph.py \
                --model_name $model_name \
                --dataset $dataset \
                --All_num_layers $la \
                --MLP_num_layers 2 \
                --heads 1 \
                --MLP_hidden 128 \
                --KHWL_num_layers 2 \
                --KHWL_hidden $hid_MLP \
                --wd $weight_decay \
                --kwl_type $khwl_type \
                --epochs $epochs \
                --runs $runs \
                --cuda $cuda \
                --lr $lr
            echo "end at $(date +%R)"
            echo "Finished training on ${dataset} with mlp ${hid_MLP} with layers ${la}"
            echo =============
          done
        done
    elif [ "$dataset" = "steam_player" ]; then
      for hid_MLP in ${list[*]}
      do
        for la in ${layers[*]}
        do
          echo =============
          echo ">>>>  Model:$model_name (default), Dataset: ${dataset} with mlp ${hid_MLP} with layers ${la}"
          echo "start at $(date +%R)"
          python main_graph.py \
              --model_name $model_name \
              --dataset $dataset \
              --All_num_layers $la \
              --MLP_num_layers 2 \
              --heads 1 \
              --MLP_hidden 128 \
              --KHWL_num_layers 2 \
              --KHWL_hidden $hid_MLP \
              --wd $weight_decay \
              --kwl_type $khwl_type \
              --epochs $epochs \
              --runs $runs \
              --cuda $cuda \
              --lr $lr
          echo "end at $(date +%R)"
          echo "Finished training on ${dataset} with mlp ${hid_MLP} with layers ${la}"
          echo =============
        done
      done
    elif [ "$dataset" = "twitter_friend" ]; then
      for hid_MLP in ${list[*]}
      do
        for la in ${layers[*]}
        do
          echo =============
          echo ">>>>  Model:$model_name (default), Dataset: ${dataset} with mlp ${hid_MLP} with layers ${la}"
          echo "start at $(date +%R)"
          python main_graph.py \
              --model_name $model_name \
              --dataset $dataset \
              --All_num_layers $la \
              --MLP_num_layers 2 \
              --heads 1 \
              --MLP_hidden 128 \
              --KHWL_num_layers 2 \
              --KHWL_hidden $hid_MLP \
              --wd $weight_decay \
              --kwl_type $khwl_type \
              --epochs $epochs \
              --runs $runs \
              --cuda $cuda \
              --lr $lr
          echo "end at $(date +%R)"
          echo "Finished training on ${dataset} with mlp ${hid_MLP} with layers ${la}"
          echo =============
        done
      done
    elif [ "$dataset" = "IMDB_dir_form" ]; then
      for hid_MLP in ${list[*]}
      do
        for la in ${layers[*]}
        do
          echo =============
          echo ">>>>  Model:$model_name (default), Dataset: ${dataset} with mlp ${hid_MLP} with layers ${la}"
          echo "start at $(date +%R)"
          python main_graph.py \
              --model_name $model_name \
              --dataset $dataset \
              --All_num_layers $la \
              --MLP_num_layers 2 \
              --heads 1 \
              --MLP_hidden 128 \
              --KHWL_num_layers 2 \
              --KHWL_hidden $hid_MLP \
              --wd $weight_decay \
              --kwl_type $khwl_type \
              --epochs $epochs \
              --runs $runs \
              --cuda $cuda \
              --lr $lr
          echo "end at $(date +%R)"
          echo "Finished training on ${dataset} with mlp ${hid_MLP} with layers ${la}"
          echo =============
        done
      done
    elif [ "$dataset" = "IMDB_dir_genre" ]; then
      for hid_MLP in ${list[*]}
      do
        for la in ${layers[*]}
        do
          echo =============
          echo ">>>>  Model:$model_name (default), Dataset: ${dataset} with mlp ${hid_MLP} with layers ${la}"
          echo "start at $(date +%R)"
          python main_graph.py \
              --model_name $model_name \
              --dataset $dataset \
              --All_num_layers $la \
              --MLP_num_layers 2 \
              --heads 1 \
              --MLP_hidden 128 \
              --KHWL_num_layers 2 \
              --KHWL_hidden $hid_MLP \
              --wd $weight_decay \
              --kwl_type $khwl_type \
              --epochs $epochs \
              --runs $runs \
              --cuda $cuda \
              --lr $lr
          echo "end at $(date +%R)"
          echo "Finished training on ${dataset} with mlp ${hid_MLP} with layers ${la}"
          echo =============
        done
      done
    elif [ "$dataset" = "IMDB_wri_form" ]; then
      for hid_MLP in ${list[*]}
      do
        for la in ${layers[*]}
        do
          echo =============
          echo ">>>>  Model:$model_name (default), Dataset: ${dataset} with mlp ${hid_MLP} with layers ${la}"
          echo "start at $(date +%R)"
          python main_graph.py \
              --model_name $model_name \
              --dataset $dataset \
              --All_num_layers $la \
              --MLP_num_layers 2 \
              --heads 1 \
              --MLP_hidden 128 \
              --KHWL_num_layers 2 \
              --KHWL_hidden $hid_MLP \
              --wd $weight_decay \
              --kwl_type $khwl_type \
              --epochs $epochs \
              --runs $runs \
              --cuda $cuda \
              --lr $lr
          echo "end at $(date +%R)"
          echo "Finished training on ${dataset} with mlp ${hid_MLP} with layers ${la}"
          echo =============
        done
      done
    elif [ "$dataset" = "IMDB_wri_genre" ]; then
      for hid_MLP in ${list[*]}
      do
        for la in ${layers[*]}
        do
          echo =============
          echo ">>>>  Model:$model_name (default), Dataset: ${dataset} with mlp ${hid_MLP} with layers ${la}"
          echo "start at $(date +%R)"
          python main_graph.py \
              --model_name $model_name \
              --dataset $dataset \
              --All_num_layers $la \
              --MLP_num_layers 2 \
              --heads 1 \
              --MLP_hidden 128 \
              --KHWL_num_layers 2 \
              --KHWL_hidden $hid_MLP \
              --wd $weight_decay \
              --kwl_type $khwl_type \
              --epochs $epochs \
              --runs $runs \
              --cuda $cuda \
              --lr $lr
          echo "end at $(date +%R)"
          echo "Finished training on ${dataset} with mlp ${hid_MLP} with layers ${la}"
          echo =============
        done
      done
      fi
done

echo "Finished all training for $model_name!"
