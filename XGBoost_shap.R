#######################################################################

library(readxl)     
library(xgboost)    
library(shapviz)    
library(ggplot2)    
library(forecast)   
library(writexl)    

sheet_names <- c("Acre","Alagoas","Amapá","Amazonas","Bahia","Ceará",
                 "Distrito Federal","Espírito Santo","Goiás","Maranhão","Mato Grosso do Sul","Mato Grosso",
                 "Minas Gerais","Pará","Paraíba","Paraná","Pernambuco","Piauí",
                 "Rio de Janeiro","Rio Grande do Norte","Rio Grande do Sul","Rondônia","Roraima","Santa Catarina",
                 "São Paulo","Sergipe","Tocantins")


param_grid <- expand.grid(
  max_depth = c(3, 6, 9),            
  eta = c(0.01, 0.1, 0.3),           
  gamma = c(0, 0.1, 0.5),            
  subsample = c(0.8, 1),             
  colsample_bytree = c(0.8, 1)       
)
colnames(param_grid) <- c("max_depth", "eta", "gamma", "subsample", "colsample_bytree")  



all_states_valid_results <- list()  

for (i in 1:length(sheet_names)) {
  
  
  data <- read_excel("Data_with_all_variables.xlsx", sheet = sheet_names[i])
  colnames(data) <- c("Data", "Case", "Tavg", "Prec", "Tdew", "Mslp", "Tmin", "Tmax", "Rehu", "Wind")
  
  
  data$Case <- log1p(data$Case)
  
  
  inTrain <- sample(1:nrow(data), size = round(nrow(data) * 0.8), replace = FALSE)
  traindata <- data[inTrain, ]
  testdata <- data[-inTrain, ]
  
  
  current_state_valid_list <- list()  
  
 
  for (j in 1:nrow(param_grid)) {
    params <- as.list(param_grid[j, ])
    
   
    model <- xgboost(
      data = as.matrix(traindata[, -(1:2)]),  # 去掉 Data, Case 两列
      label = traindata$Case,
      params = params,
      nrounds = 100,
      objective = "reg:squarederror"
    )
    
    
    train_rmse <- model$evaluation_log$train_rmse[model$niter]          
    train_sd <- sd(traindata$Case)                                     
    test_pred <- predict(model, as.matrix(testdata[, -(1:2)]))         
    test_RMSE <- accuracy(test_pred, testdata$Case)[2]                 
    test_sd <- sd(testdata$Case)                                       
    
   
    is_train_good <- (train_rmse < 0.5 * train_sd)
    is_test_good <- (test_RMSE < 0.5 * test_sd)
    is_test_ok <- (test_RMSE > 0.5 * test_sd & test_RMSE < 1 * test_sd)
    
    
    if ((is_train_good & is_test_good) | (is_train_good & is_test_ok)) {
      current_rmse <- test_RMSE
      
      param_with_results <- data.frame(
        State = sheet_names[i],
        max_depth = params$max_depth,
        eta = params$eta,
        gamma = params$gamma,
        subsample = params$subsample,
        colsample_bytree = params$colsample_bytree,
        RMSE = current_rmse,
        is_train_good = is_train_good,
        is_test_good = is_test_good,
        is_test_ok = is_test_ok,
        stringsAsFactors = FALSE
      )
      
      
      current_state_valid_list <- append(current_state_valid_list, list(param_with_results))
    }
  }
  
  
  if (length(current_state_valid_list) > 0) {
    all_states_valid_results[[i]] <- do.call(rbind, current_state_valid_list)
  } else {
    
    all_states_valid_results[[i]] <- data.frame(
      State = character(0),               
      max_depth = numeric(0),
      eta = numeric(0),
      gamma = numeric(0),
      subsample = numeric(0),
      colsample_bytree = numeric(0),
      RMSE = numeric(0),
      is_train_good = logical(0),
      is_test_good = logical(0),
      is_test_ok = logical(0),
      stringsAsFactors = FALSE
    )
  }
  
 
  if (!is.null(all_states_valid_results[[i]]) && nrow(all_states_valid_results[[i]]) > 0) {
    valid_df <- all_states_valid_results[[i]]  
    best_idx <- which.min(valid_df$RMSE)       
    best_row <- valid_df[best_idx, ]           
    
    best_params <- as.list(best_row[, c("max_depth", "eta", "gamma", "subsample", "colsample_bytree")])  
    
    
    
    
    final_model <- xgboost(
      data = as.matrix(traindata[, -(1:2)]),
      label = traindata$Case,
      params = best_params,
      nrounds = 100,
      objective = "reg:squarederror"
    )
    
    
    saveRDS(final_model, file = paste0(sheet_names[i], "_xgboost_model_best.rds"))
    
    
    shap_final <- shapviz(final_model, X_pred = as.matrix(traindata[, -(1:2)]))
    p_shap_importance <- sv_importance(shap_final) + theme_bw()
    p_shap_beeswarm <- sv_importance(shap_final, kind = "beeswarm") + theme_bw()
    ggsave(paste0(sheet_names[i], "_shap_importance_best.png"), p_shap_importance, device = "png", dpi = 300, width = 8, height = 6)
    ggsave(paste0(sheet_names[i], "_shap_beeswarm_best.png"), p_shap_beeswarm, device = "png", dpi = 300, width = 8, height = 6)
    
    
    
  } 
}


