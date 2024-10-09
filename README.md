# GADE+: A Graph-based Anchor-enhanced Framework for Targeted Document Detection

## Dependencies

* Compatible with Python 3.7.16
* Dependencies can be installed using requirements.txt


### Datasets
We construct four labeled datasets for the targeted document detection task, i.e., `Wiki-100`, `Wiki-200`, `Wiki-300`, and `Web-Test`. The former three datasets are
constructed from Wikipedia and the last dataset is constructed from Web documents.

* The four labeled datasets `Wiki-100`, `Wiki-200`, `Wiki-300`, and `Web-Test` are placed in the `datasets` folder. Please unzip `Wiki100.zip`, `Wiki200.zip`, `Wiki300.zip`, and `Web_Test.zip` under `datasets/`.

### Usage

##### Run the main code (**GADE+**):
  
* python main_Wiki.py --gpu 1 --gcn_layer 2 --anchor_nums 8 --num_pers 4 --epochs 10 --dropout 0.5 --p2a_coff 5e-2 --hid_dim 256 --data_type Wiki100
  --model_name GADE_100 --p2a_regularization --tau 1.0 --aux_ce_reg --lamda 1.0
  
* python main_Wiki.py --gpu 1 --gcn_layer 2 --anchor_nums 16 --num_pers 4 --epochs 10 --dropout 0.5 --p2a_coff 5e-3 --hid_dim 256 --data_type Wiki200
  --model_name GADE_200 --p2a_regularization --tau 1.0 --aux_ce_reg --lamda 1.0

* python main_Wiki.py --gpu 1 --gcn_layer 2 --anchor_nums 8 --num_pers 4 --epochs 10 --dropout 0.5 --weight_decay 5e-2 --p2a_coff 1e-3 --hid_dim 512
  --data_type Wiki300 --model_name GADE_300 --p2a_regularization --tau 1.0 --aux_ce_reg --lamda 1.0


##### Test the model's performance on Web-Test dataset:

-- For **GADE+**:

* python main_Web_Test.py --gpu 1 --gcn_layer 2 --anchor_nums 8 --num_pers 4 --epochs 10 --dropout 0.5 --weight_decay 5e-2 --p2a_coff 1e-3
  --hid_dim 512 --data_type Wiki300 --model_name GADE_300 --p2a_regularization --tau 1.0 --aux_ce_reg --lamda 1.0
