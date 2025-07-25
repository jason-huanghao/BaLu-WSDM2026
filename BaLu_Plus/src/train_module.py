import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# import time
import copy
# import logging
from torch_geometric.data import Data
from src.data_utils import create_data_edge_index, update_observed_mask, mask_edge
from src.train_utils import compute_imputation_loss, compute_balance_loss, compute_treatment_loss, compute_outcome_loss
from src.train_utils import EarlyStopping


class ModelTrainer:
    def __init__(self, model_class, params, device=None, seed=42):
        self.model_class = model_class
        self.params = params

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Data
        self.norm_y = self.params.get('norm_y', True)
        self.edge_dropout = self.params.get('rel_dropout', 0.0)
        self.y_mean = 0.0
        self.y_std = 0.0

        # Model
        self.model = None
        self.history = {'epoch': [], 'train_loss': [], 'val_loss': []} # For basic loss tracking
        self.loss_funs = {"t_loss": compute_treatment_loss,
                          "y_loss": compute_outcome_loss,
                          "b_loss": compute_balance_loss,
                          "impute_loss": compute_imputation_loss}
        self.loss_weights = {"y_loss": torch.tensor(self.params.get('alpha', 1.0), device=self.device),
                        "t_loss": torch.tensor(self.params.get('beta', 1e-4), device=self.device),
                        "b_loss": torch.tensor(self.params.get('gamma', 1e-4), device=self.device),
                        "impute_loss": torch.tensor(self.params.get('eta', 1e-4), device=self.device)}
    
    def train(self, data):
        # Extract training parameters
        lr = self.params.get('lr', 1e-3)
        weight_decay = self.params.get('weight_decay', 1e-4)
        n_epochs = self.params.get('n_epochs', 2000)

        self.model = self._create_model(data, self.params).to(self.device)        # vars convert a Namespace to dict
        data_device = self._prepare_data(data)

        # history = {'epoch': [],
        #            'train_loss': [],
        #            'val_loss': [],
        #            'train_metrics': [],
        #            'val_metrics': []}
        # log_interval = 20
        # print("created model, load the data!")

        early_stop = self.params.get("early_stop", True)
        if early_stop:
            early_stopping = EarlyStopping(
                patience=self.params.get("patience", 25),
                verbose=False) # Uncomment for more detailed ES output
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        tr_ratio, val_ratio = torch.sum(data_device.train_mask).float(), torch.sum(data_device.val_mask).float()
        tr_ratio, val_ratio = tr_ratio / (tr_ratio+val_ratio), val_ratio / (tr_ratio+val_ratio)
        
        for epoch in range(n_epochs):
            self.model.train()
            optimizer.zero_grad()

            # print("input")
            outputs = self.model(data_device)

            # print("output")
            train_loss_dict = self._compute_loss(outputs, data_device, unit_indexes=data_device.train_mask)
            train_loss = self._compute_total_loss(train_loss_dict)
            train_loss.backward()
            optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                val_loss_dict = self._compute_loss(outputs, data_device, unit_indexes=data_device.val_mask)
                if 'b_loss' in val_loss_dict:
                    val_loss_dict['b_loss'] = train_loss_dict['b_loss']*tr_ratio + val_loss_dict['b_loss']*val_ratio
                # if 'impute_loss' in val_loss_dict:
                #     val_loss_dict['impute_loss'] = train_loss_dict['impute_loss']*tr_ratio + val_loss_dict['impute_loss']*val_ratio

                val_loss = self._compute_total_loss(val_loss_dict)

            # history['epoch'].append(epoch)
            # history['train_loss'].append(train_loss.item())
            # history['val_loss'].append(val_loss.item())
            
            # if (epoch + 1) % log_interval == 0:
            #     elapsed = time.time() - start_time
            #     print(f"Epoch {epoch+1}/{n_epochs} [{elapsed:.2f}s]")
            #     train_losses = {k: round(float(v), 4) for k, v in train_loss_dict.items()}
            #     val_losses = {k: round(float(v), 4) for k, v in val_loss_dict.items()}
            #     print(f"Train loss: {train_loss.item():.4f}, Detailed: {train_losses}")
            #     print(f"Valid loss: {val_loss.item():.4f}, Detailed: {val_losses}")
            log_interval = 20
            if (epoch + 1) % log_interval == 0:
                print(f"train_loss: {train_loss.item()}; val_loss: {val_loss.item()}")

            # Early stopping
            if early_stop:
                early_stopping(val_loss.item(), self.model)
                if early_stopping.early_stop:
                    print("Early stopping!")
                    break
        
        return early_stopping.get_best_model(self.model)
    
        # self.history = history
        # self._plot_loss_curves()
        # return self.model


    def evaluate(self, data, unit_indexes=None, on_complete_test=False):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        self.model.eval()
        data_device = self._prepare_data(data)
        return self._evaluate(self.model, data_device, unit_indexes=unit_indexes, on_complete_test=on_complete_test)
    
    def _evaluate(self, model, data, unit_indexes=None, on_complete_test=False):
        model.eval()
        results = {}

        if data.x.device.type != self.device.type:
            data = self._prepare_data(data)
        data = copy.deepcopy(data)          # !!!! do not modify the data, better to use copy
        
        if on_complete_test:    # testing on test set with complete data 
            # update observed mask, and corresponding tensors (all should in device)
            observed_mask = update_observed_mask(data.test_mask, data.observed_mask, data.n_units, data.n_attrs)
            data_edge_index, data_edge_attr = create_data_edge_index(data.arr_X, data.n_units, data.n_attrs)
            observed_mask = observed_mask.to(self.device)
            data_edge_index, data_edge_attr = data_edge_index.to(self.device), data_edge_attr.to(self.device)
            
            data_edge_mask = torch.cat((observed_mask, observed_mask), dim=0)     # i -> j and j -> i
            obs_data_edge_index, obs_data_edge_attr = mask_edge(data_edge_index, data_edge_attr, data_edge_mask, True)
            
            data.observed_mask = observed_mask
            data.edge_index, data.edge_attr = obs_data_edge_index, obs_data_edge_attr
            
        with torch.no_grad():
            outputs = model(data)

            if self.norm_y:
                outputs['pred_outcomes'] = outputs['pred_outcomes'] * self.y_std + self.y_mean
                outputs['pred_outcomes_0'] = outputs['pred_outcomes_0'] * self.y_std + self.y_mean
                outputs['pred_outcomes_1'] = outputs['pred_outcomes_1'] * self.y_std + self.y_mean
            outputs['estimated_effect'] = outputs['pred_outcomes_1'] - outputs['pred_outcomes_0']

            est_effects = outputs.get('estimated_effect')
            true_effects = getattr(data, 'true_effect')
            
            pehe_ts = torch.sqrt(F.mse_loss(true_effects[unit_indexes], est_effects[unit_indexes])).item() if unit_indexes is not None else torch.sqrt(F.mse_loss(true_effects, est_effects)).item()
            mae_ate_ts = torch.abs(torch.mean(true_effects[unit_indexes]) - torch.mean(est_effects[unit_indexes])).item() if unit_indexes is not None else torch.abs(torch.mean(true_effects) - torch.mean(est_effects)).item()
            results['effect_pehe'] = pehe_ts
            results['effect_mae'] = mae_ate_ts
        return results
    
    def _compute_total_loss(self, loss_dict):
        total_loss = torch.tensor(0.0, device=self.device)
        for loss_name, loss_weight in self.loss_weights.items():
            if loss_name in loss_dict:
                total_loss += loss_weight * loss_dict[loss_name]
        return total_loss
    
    def _compute_loss(self, outputs, data, unit_indexes=None):
        loss_dict = {}
        for loss_name in self.loss_funs.keys():
            loss_dict[loss_name] = self.loss_funs[loss_name](outputs, data, unit_indexes=unit_indexes)
        return loss_dict
    
    def _create_model(self, data, args):
        model = self.model_class(data, args)
        return model
    
    def _prepare_data(self, data):
        data_device = self._to_device(data)    # to GPU

        if self.norm_y:
            self.y_mean, self.y_std = torch.mean(data_device.outcome), torch.std(data_device.outcome)
            data_device.outcome = (data_device.outcome - self.y_mean) / self.y_std
        return data_device

    def _to_device(self, data):
        data_device = Data()   
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                setattr(data_device, key, value.to(self.device))
            else:
                setattr(data_device, key, value)
        return data_device
    
