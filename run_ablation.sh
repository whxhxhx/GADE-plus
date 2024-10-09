# Wiki100
python main_new.py --gpu 1 --gcn_layer 2 --anchor_nums 8 --num_pers 4 --epochs 10 --dropout 0.5 --p2a_coff 5e-2 --hid_dim 256 --data_type Wiki100 --model_name GADE_100 --p2a_regularization --tau 1.0
python main_new.py --gpu 1 --gcn_layer 2 --anchor_nums 8 --num_pers 4 --epochs 10 --dropout 0.5 --p2a_coff 5e-2 --hid_dim 256 --data_type Wiki100 --model_name GADE_100 --aux_ce_reg --lamda 1.0
python main_new.py --gpu 1 --gcn_layer 2 --anchor_nums 8 --num_pers 4 --epochs 10 --dropout 0.5 --p2a_coff 5e-2 --hid_dim 256 --data_type Wiki100 --model_name GADE_100

# Wiki200
python main_new.py --gpu 1 --gcn_layer 2 --anchor_nums 16 --num_pers 4 --epochs 10 --dropout 0.5 --p2a_coff 5e-3 --hid_dim 256 --data_type Wiki200 --model_name GADE_200 --p2a_regularization --tau 1.0
python main_new.py --gpu 1 --gcn_layer 2 --anchor_nums 16 --num_pers 4 --epochs 10 --dropout 0.5 --p2a_coff 5e-3 --hid_dim 256 --data_type Wiki200 --model_name GADE_200 --aux_ce_reg --lamda 1.0
python main_new.py --gpu 1 --gcn_layer 2 --anchor_nums 16 --num_pers 4 --epochs 10 --dropout 0.5 --p2a_coff 5e-3 --hid_dim 256 --data_type Wiki200 --model_name GADE_200

# Wiki300
python main_new.py --gpu 1 --gcn_layer 2 --anchor_nums 8 --num_pers 4 --epochs 10 --dropout 0.5 --weight_decay 5e-2 --p2a_coff 1e-3 --hid_dim 512 --data_type Wiki300 --model_name GADE_300 --p2a_regularization --tau 1.0
python main_new.py --gpu 1 --gcn_layer 2 --anchor_nums 8 --num_pers 4 --epochs 10 --dropout 0.5 --weight_decay 5e-2 --p2a_coff 1e-3 --hid_dim 512 --data_type Wiki300 --model_name GADE_300 --aux_ce_reg --lamda 1.0
python main_new.py --gpu 1 --gcn_layer 2 --anchor_nums 8 --num_pers 4 --epochs 10 --dropout 0.5 --weight_decay 5e-2 --p2a_coff 1e-3 --hid_dim 512 --data_type Wiki300 --model_name GADE_300

# Web_Test
python main_Web_Test.py --gpu 1 --gcn_layer 2 --anchor_nums 8 --num_pers 4 --epochs 10 --dropout 0.5 --weight_decay 5e-2 --p2a_coff 1e-3 --hid_dim 512 --model_name GADE_300_wo_aux_ce --p2a_regularization --tau 1.0
python main_Web_Test.py --gpu 1 --gcn_layer 2 --anchor_nums 8 --num_pers 4 --epochs 10 --dropout 0.5 --weight_decay 5e-2 --p2a_coff 1e-3 --hid_dim 512 --model_name GADE_300_wo_a_comp --aux_ce_reg --lamda 1.0
python main_Web_Test.py --gpu 1 --gcn_layer 2 --anchor_nums 8 --num_pers 4 --epochs 10 --dropout 0.5 --weight_decay 5e-2 --p2a_coff 1e-3 --hid_dim 512 --model_name GADE_300_wo_aux_ce_a_comp
