###### PART 3

X_train = text_transformer.fit_transform(df['Text'])
X_test = text_transformer.fit_transform(df_test['Text'])
Y_train = testDF['Category']


#Selecting 4 different values for batch size
nn_001 = MLPClassifier(hidden_layer_sizes=(1,5), activation='identity', solver='sgd', batch_size='auto', learning_rate='constant', max_iter=100, random_state=42, learning_rate_init=0.001)
nn_01 = MLPClassifier(hidden_layer_sizes=(1,5), activation='identity', solver='sgd', batch_size='auto', learning_rate='constant', max_iter=100, random_state=42, learning_rate_init=0.01)
nn_1 = MLPClassifier(hidden_layer_sizes=(1,5), activation='identity', solver='sgd', batch_size='auto', learning_rate='constant', max_iter=100, random_state=42, learning_rate_init=0.1)
nn_2 = MLPClassifier(hidden_layer_sizes=(1,5), activation='identity', solver='sgd', batch_size='auto', learning_rate='constant', max_iter=100, random_state=42, learning_rate_init=0.2)


learning_rates = [nn_001, nn_01, nn_1, nn_2]
kf = KFold(n_splits=5)

training_acc_avg = []
validation_acc_avg = []
for learning_rate in learning_rates:
    training_acc = []
    validation_acc = []
    for train_indices, test_indices in kf.split(X_train):
        learning_rate.fit(X_train[train_indices], Y_train[train_indices])
        training_acc.append(learning_rate.score(X_train[train_indices], Y_train[train_indices]))
        validation_acc.append(learning_rate.score(X_train[test_indices], Y_train[test_indices]))
    
    training_acc_avg.append(statistics.mean(training_acc))
    validation_acc_avg.append(statistics.mean(validation_acc))



print("Training accuracy:", training_acc_avg, "\nValidation accuracy:", validation_acc_avg)

testing_acc_nn = []
for learning_rate in learning_rates:
    learning_rate.fit(X_train, y_train)
    testing_acc_nn.append(metrics.accuracy_score(y_test, learning_rate.predict(X_test)))


print("Testing accuracy:", testing_acc_nn)




plt.plot([0.001, 0.01, 0.1, 0.2], training_acc_avg, label='Average Training Accuracy')
plt.plot([0.001, 0.01, 0.1, 0.2], validation_acc_avg, label='Average Validation Accuracy')
plt.plot([0.001, 0.01, 0.1, 0.2], testing_acc_nn, label='Testing Accuracy')
plt.title('Accuracy of Training, Validation (5-fold cross validated) and Testing accuracy for differing learning rates')
plt.xlabel('Initial learning rate')
plt.ylabel('Accuracy')
plt.legend()