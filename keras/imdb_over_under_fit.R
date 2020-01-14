# https://keras.rstudio.com/articles/tutorial_overfit_underfit.html

library(keras)
library(dplyr)
library(ggplot2)
library(tidyr)
library(tibble)

num_words <- 10000
imdb <- dataset_imdb(num_words = num_words)

c(train_data, train_labels) %<-% imdb$train
c(test_data, test_labels) %<-% imdb$test

multi_hot_sequences <- function(sequences, dimension) {
  multi_hot <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences)) {
    multi_hot[i, sequences[[i]]] <- 1
  }
  multi_hot
}

train_data <- multi_hot_sequences(train_data, num_words)
test_data <- multi_hot_sequences(test_data, num_words)

first_text <- data.frame(word = 1:10000, value = train_data[1, ])
ggplot(first_text, aes(x = word, y = value)) +
  geom_line() +
  theme(axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank())

baseline_model <-
  keras_model_sequential() %>%
  layer_dense(units = 16, activation = 'relu', input_shape = 10000) %>%
  layer_dense(units = 16, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')


baseline_model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = list('accuracy')
)

baseline_model %>% summary()

baseline_history <- baseline_model %>% fit(
  train_data,
  train_labels,
  epochs = 20,
  batch_size = 512,
  validation_data = list(test_data, test_labels),
  verbose = 2
)

smaller_model <-
  keras_model_sequential() %>%
  layer_dense(units = 4, activation = 'relu', input_shape = 10000) %>%
  layer_dense(units = 4, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')

smaller_model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = list('accuracy')
)

smaller_model %>% summary()

smaller_history <- smaller_model %>% fit(
  train_data,
  train_labels,
  epochs = 20,
  batch_size = 512,
  validation_data = list(test_data, test_labels),
  verbose = 2
)

bigger_model <-
  keras_model_sequential() %>%
  layer_dense(units = 512, activation = 'relu', input_shape = 10000) %>%
  layer_dense(units = 512, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')

bigger_model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = list('accuracy')
)

bigger_model %>% summary()

bigger_history <- bigger_model %>% fit(
  train_data,
  train_labels,
  epochs = 20,
  batch_size = 512,
  validation_data = list(test_data, test_labels),
  verbose = 2
)

