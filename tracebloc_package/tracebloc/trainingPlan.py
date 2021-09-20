import requests,json

class TrainingPlan:

	"""
		creating a training plan and assign data set
		parameters: modelId, datasetId, token

		methods:get_parameters, get_trainingplan
		"""

	def __init__(self,modelId,datasetId,token):

		self.__url = 'https://xray-backend-develop.azurewebsites.net/'
		# self.__url = 'http://127.0.0.1:8000/'
		self.__token = token
		self.__message = 'training'
		self.__datasetId = datasetId
		self.__epochs = 10
		self.__cycles = 1
		self.__modelName = modelId
		self.__optimizer = "adam"
		self.__lossFunction = "categorical_crossentropy"
		self.__learningRate = 0.001
		self.__stepsPerEpoch = "None"
		self.__initialEpoch = 0
		self.__validationSteps = "None"
		self.__batchSize = 32
		self.__featurewise_center = False
		self.__samplewise_center = False
		self.__featurewise_std_normalization = False
		self.__samplewise_std_normalization = False
		self.__zca_whitening = False
		self.__rotation_range = 0
		self.__width_shift_range = 0.0
		self.__height_shift_range = 0.0
		self.__brightness_range = "None"
		self.__shear_range = 0.0
		self.__zoom_range = 0.0
		self.__channel_shift_range = 0.0
		self.__fill_mode = "nearest"
		self.__cval = 0.0
		self.__horizontal_flip = False
		self.__vertical_flip = False
		self.__rescale = "None"
		self.__data_format = "None"
		self.__validation_split = 0.1
		self.__dtype = "None"
		self.__shuffle = True
		self.__layers_non_trainable = ''
		self.__metrics = str(["accuracy"])
		self.__objective = ""
		self.__name = ""
		self.__modelType = ""
		self.__category = ""
		self.__upperboundTime = 0

		header = {'Authorization' : f"Token {self.__token}"}
		re = requests.post(f"{self.__url}check-model/",headers= header,data={'datasetId':self.__datasetId,'modelName':self.__modelName})
		body_unicode = re.content.decode('utf-8')
		content = json.loads(body_unicode)
		if content["status"] == "failed":
			print("Please ensure that modelid provided have same output clases as in dataset \nand input shape match the below condition:\nFor medical--> (None, 224, 224, 3)\nFor industrial--> (None, 48, 48, 3)")
		else:
			print("Model and dataset selected, please set parameters")

	def setExperimentCategory(self,category:str):
		'''
		String.
		Category of experiment, like classification
		example:setExperimentCategory('classification')
		'''
		if type(category) == str:
			self.__category = category
		else:
			print("Invalid input type given")

	def setModelType(self,modelType:str):
		'''
		String.
		Type of model used in the experiment, like VGGNET
		example:setModelType('VGGNET')
		'''
		if type(modelType) == str:
			self.__modelType = modelType
		else:
			print("Invalid input type given")

	def setExperimentName(self,name:str):
		'''
		String.
		Name of the experiment
		example:setExperimentName('Classifying manufacturing defects')
		'''
		if type(name) == str:
			self.__name = name
		else:
			print("Invalid input type given")

	def setExperimentObjective(self,objective:str):
		'''
		String.
		Objective of the experiment
		example:setExperimentObjective('Classify images using Convolutional Neural Networks (specifically, VGG16) pre-trained on the ImageNet dataset with Keras deep learning library.')
		'''
		if type(objective) == str:
			self.__objective = objective
		else:
			print("Invalid input type given")

	def setEpochs(self,epochs:int):
		'''
		Integer. 
		Number of epochs to train the model.
		An epoch is an iteration over the entire data provided.
		Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch".
		The model is not trained for a number of iterations given by epochs,
		but merely until the epoch of index epochs is reached.
		example: setEpochs(100)
		default: 10
		'''
		if type(epochs) == int and epochs!=0:
			self.__epochs = epochs
		else:
			print("Invalid input type or value '0' given")

	def setCycles(self,cycles:int):
		'''
		Set number of cycles
		parameters: integer type values.
		example: setCycles(10)
		default: 1
		'''
		if type(cycles) == int and cycles!=0:
			self.__cycles = cycles
		else:
			print("Invalid input type or value '0' given")

	def setOptimizer(self,optimizer:str):
		'''
		Set optimizer
		parameters: string type values.
		example: setOptimizer('rmsprop')
		supported optimizers: ['adam','rmsprop','sgd','adadelta', 'adagrad', 'adamax','nadam', 'ftrl']
		default: 'adam'
		'''
		o = ['adam','rmsprop','sgd',
		'adadelta', 'adagrad', 'adamax',
		'nadam', 'ftrl']
		try:
			o.index(optimizer.lower())
			self.__optimizer = optimizer.lower()
		except:
			print(f"Please provide supported optimizers: {o}")

	def setLossFunction(self,lossFunction:str):
		'''
		Set loss function
		parameters: string type values.
		example: setLossFunction('binary_crossentropy')
		supported loss functions: ['binary_crossentropy','categorical_crossentropy']
		default: "categorical_crossentropy"
		'''
		l = ['binary_crossentropy','categorical_crossentropy']
		try:
			l.index(lossFunction.lower())
			self.__lossFunction = lossFunction.lower()
		except:
			print(f"Please provide supported loss functions: {l}")


	def setLearningrate(self,learningRate:float):
		'''
		Set learning rate
		parameters: float type values.
		example: setLearningrate(0.0001)
		default: 0.001
		'''
		if type(learningRate) == float and learningRate!=0:
			self.__learningRate = learningRate
		else:
			print("Invalid input type or value '0' given")

	def setStepsPerEpoch(self,stepsPerEpoch:int):
		'''
		 Integer.
		 Total number of steps (batches of samples) before declaring
		 one epoch finished and starting the next epoch.
		 example: setStepsPerEpoch(5)
		 default: None
		'''
		if type(stepsPerEpoch) == int and stepsPerEpoch!=0:
			self.__stepsPerEpoch = stepsPerEpoch
		else:
			print("Invalid input type or value '0' given")

	def setInitialEpoch(self,initialEpoch:int):
		'''
		Integer. Epoch at which to start training 
		(useful for resuming a previous training run).
		example: setInitialEpoch(2)
		default: 0
		'''
		if type(initialEpoch) == int:
			self.__initialEpoch = initialEpoch
		else:
			print("Invalid input type given")

	def setValidationSteps(self,validationSteps:int):
		'''
		Integer. Total number of steps (batches of samples) to draw before stopping 
		when performing validation at the end of every epoch.
		example: setValidationSteps(10)
		default: None
		'''
		if type(validationSteps) == int and validationSteps!=0:
			self.__validationSteps = validationSteps
		else:
			print("Invalid input type or value '0' given")

	def setBatchSize(self,batchSize:int):
		'''
		Integer. 
		Number of samples per gradient update.
		example: setBatchSize(16)
		default: 32
		'''
		if type(batchSize) == int:
			self.__batchSize = batchSize
		else:
			print("Invalid input type given")

	def setFeaturewiseCenter(self,featurewise_center:bool):
		'''
		Boolean.
		Set input mean to 0 over the dataset, feature-wise.
		example: setBatchSize(True)
		default: False
		'''
		if type(featurewise_center) == bool:
			self.__featurewise_center = featurewise_center
		else:
			print("Invalid input type given")

	def setSamplewiseCenter(self,samplewise_center:bool):
		'''
		Boolean. 
		Set each sample mean to 0.
		example: setSamplewiseCenter(True)
		default: False
		'''
		if type(samplewise_center) == bool:
			self.__samplewise_center = samplewise_center
		else:
			print("Invalid input type given")

	def setFeaturewiseStdNormalization(self,featurewise_std_normalization:bool):
		'''
		Boolean. 
		Divide inputs by std of the dataset, feature-wise.
		example: setFeaturewiseStdNormalization(True)
		default: False
		'''
		if type(featurewise_std_normalization) == bool:
			self.__featurewise_std_normalization = featurewise_std_normalization
		else:
			print("Invalid input type given")

	def setSamplewiseStdNormalization(self,samplewise_std_normalization:bool):
		'''
		Boolean. 
		Divide each input by its std.
		example: setSamplewiseStdNormalization(True)
		default: False
		'''
		if type(samplewise_std_normalization) == bool:
			self.__samplewise_std_normalization = samplewise_std_normalization
		else:
			print("Invalid input type given")

	def setZcaWhitening(self,zca_whitening:bool):
		'''
		Boolean. 
		Apply ZCA whitening.
		example: setZcaWhitening(True)
		default: False
		'''
		if type(zca_whitening) == bool:
			self.__zca_whitening = zca_whitening
		else:
			print("Invalid input type given")

	def setRotationRange(self,rotation_range:int):
		'''
		Integer. 
		Degree range for random rotations.
		example: setRotationRange(2)
		default: 0
		'''
		if type(rotation_range) == int:
			self.__rotation_range = rotation_range
		else:
			print("Invalid input type given")

	def setWidthShiftRange(self,width_shift_range:float):
		'''
		Float.
		Fraction of total width, if < 1, or pixels if >= 1. 
		example: setWidthShiftRange(0.4)
		default: 0.0
		'''
		if type(width_shift_range) == float:
			self.__width_shift_range = width_shift_range
		else:
			print("Invalid input type given")

	def setHeightShiftRange(self,height_shift_range:float):
		'''
		Float.
		Fraction of total height, if < 1, or pixels if >= 1.
		example: setHeightShiftRange(0.4)
		default: 0.0
		'''
		if type(height_shift_range) == float:
			self.__height_shift_range = height_shift_range
		else:
			print("Invalid input type given")

	def setBrightnessRange(self,brightness_range:tuple):
		'''
		Tuple of two floats.
		Range for picking a brightness shift value from.
		example: setBrightnessRange((0.1,0.4))
		default: None
		'''
		if type(brightness_range)==tuple and len(brightness_range)==2:

			if type(brightness_range[0])==float and type(brightness_range[1])==float:
				brightness_range = str(brightness_range)
				self.__brightness_range = brightness_range
			else:
				print("provide float values")
		else:
			print("Please provide tuple of two floats")

	def setShearRange(self,shear_range:float):
		'''
		Float. 
		Shear Intensity (Shear angle in counter-clockwise direction in degrees)
		example: setShearRange(0.2)
		default: 0.0
		'''
		if type(shear_range) == float:
			self.__shear_range = shear_range
		else:
			print("Invalid input type given")

	def setZoomRange(self,zoom_range:float):
		'''
		Float. 
		Range for random zoom. Range selected as for float value provided,
		[lower, upper] = [1-zoom_range, 1+zoom_range].
		example: setZoomRange(0.2)
		default: 0.0
		'''
		if type(zoom_range) == float:
			self.__zoom_range = zoom_range
		else:
			print("Invalid input type given")

	def setChannelShiftRange(self,channel_shift_range:float):
		'''
		Float. 
		Range for random channel shifts.
		example: setChannelShiftRange(0.4)
		default: 0.0
		'''
		if type(channel_shift_range) == float:
			self.__channel_shift_range = channel_shift_range
		else:
			print("Invalid input type given")

	def setFillMode(self,fill_mode:str):
		'''
		String.
		One of {"constant", "nearest", "reflect" or "wrap"}. 
		Points outside the boundaries of the input are filled according to the given mode:
		- 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k) 
		- 'nearest': aaaaaaaa|abcd|dddddddd 
		- 'reflect': abcddcba|abcd|dcbaabcd 
		- 'wrap': abcdabcd|abcd|abcdabcd
		example: setFillMode("nearest")
		default: "nearest"
		'''
		f = ["constant", "nearest", "reflect", "wrap"]
		try:
			f.index(fill_mode.lower())
			self.__fill_mode = fill_mode.lower()
		except:
			print(f"Please provide supported fill modes: {f}")

	def setCval(self,cval:float):
		'''
		Float. 
		Value used for points outside the boundaries when fill_mode = "constant".
		example: setCval(0.3)
		default: 0.0
		'''
		if type(cval) == float:
			self.__cval = cval
		else:
			print("Invalid input type given")

	def setHorizontaFlip(self,horizontal_flip:bool):
		'''
		Boolean. 
		Randomly flip inputs horizontally.
		example: setHorizontaFlip(True)
		default: False
		'''
		if type(horizontal_flip) == bool:
			self.__horizontal_flip = horizontal_flip
		else:
			print("Invalid input type given")

	def setVerticalFlip(self,vertical_flip:bool):
		'''
		Boolean. 
		Randomly flip inputs vertically.
		example: setVerticalFlip(True)
		default: False
		'''
		if type(vertical_flip) == bool:
			self.__vertical_flip = vertical_flip
		else:
			print("Invalid input type given")

	def setRescale(self,rescale:int):
		'''
		Integer. 
		If 0, no rescaling is applied, 
		otherwise we multiply the data by the value provided 
		(after applying all other transformations).
		example: setRescale(2)
		default: None
		'''
		if type(rescale) == int:
			self.__rescale = rescale
		else:
			print("Invalid input type given")

	def setDataFormat(self,data_format:str):
		'''
		String. 
		Image data format, either "channels_first" or "channels_last". 
		"channels_last" mode means that the images should have shape 
		(samples, height, width, channels), 
		"channels_first" mode means that the images should have shape 
		(samples, channels, height, width).
		example: setDataFormat("channels_first")
		default: None
		'''
		d = ["channels_first", "channels_last"]
		try:
			d.index(data_format.lower())
			self.__data_format = data_format.lower()
		except:
			print(f"Please provide supported fill modes: {d}")

	def setValidationSplit(self,validation_split:float):
		'''
		Float. 
		Fraction of images reserved for validation (strictly between 0 and 1).
		example: setValidationSplit(0.2)
		default: 0.1
		'''
		if type(validation_split)==float and 0.0 <= validation_split <= 1.0:
			self.__validation_split = validation_split
		else:
			print("Invalid input type or out of range[0,1]")

	def setDtype(self,dtype:str):
		'''
		String. 
		Dtype to use for the generated arrays.
		example: setDtype('float32')
		default: None
		'''
		if type(dtype) == str:
			self.__dtype = dtype
		else:
			print("Invalid input type given")

	def setShuffle(self,shuffle:bool):
		'''
		Boolean. 
		Shuffles the image data.
		example: setShuffle(False)
		default: True
		'''
		if type(shuffle) == bool:
			self.__shuffle = shuffle
		else:
			print("Invalid input type given")

	def setLayersNotToTrain(self,layers_non_trainable:list):
		'''
		List of strings. 
		Freezes the layers name provided.
		example: setLayersNotToTrain(['conv1','fc1'])
		default: None
		'''
		if type(layers_non_trainable)== list and all(isinstance(sub, str) for sub in layers_non_trainable):
			layers_non_trainable = str(layers_non_trainable)
			self.__layers_non_trainable = layers_non_trainable
		else:
			print("Provide values as list of strings")

	# def setMetrics(self,metrics:list):
	# 	'''
	# 	List of strings. 
	# 	List of metrics to be evaluated by the model 
	# 	during training and testing.
	# 	example: setMetrics(['accuracy','mse'])
	# 	default: ['accuracy']
	# 	'''
	# 	if type(metrics)== list and all(isinstance(sub, str) for sub in metrics):
	# 		metrics = str(metrics)
	# 		self.__metrics = metrics
	# 	else:
	# 		print("Provide values as list of strings")

	def __display_time(self,seconds, granularity=5):
                intervals = (
                ('weeks', 604800),  # 60 * 60 * 24 * 7
                ('days', 86400),    # 60 * 60 * 24
                ('hours', 3600),    # 60 * 60
                ('minutes', 60),
                ('seconds', 1),)
                result = []

                for name, count in intervals:
                    value = seconds // count
                    if value:
                        seconds -= value * count
                        if value == 1:
                            name = name.rstrip('s')
                        result.append("{} {}".format(value, name))
                return ', '.join(result[:granularity])


	def getEstimate(self):

		header = {'Authorization' : f"Token {self.__token}"}
		re = requests.post(f"{self.__url}flops/",headers= header,data={'datasetId':self.__datasetId,
			'batchSize':self.__batchSize,'noOfEpochs':self.__epochs,'modelName':self.__modelName})
# 		print(re.status_code)
		if re.status_code == 200:

			body_unicode = re.content.decode('utf-8')
			content = int(json.loads(body_unicode))
			self.__upperboundTime = content
			cycleTime = content * self.__cycles
			display = self.__display_time(cycleTime)

			print(f"It will take around {display} to complete {self.__cycles} cycles for given training plan.")


	def create(self):
		#Create Experiment   
		header = {'Authorization' : f"Token {self.__token}"}
		re = requests.post(f"{self.__url}experiments/",headers= header,data=self.__getParameters())
		# print(re.status_code)
		if re.status_code == 201:

			print("TrainingPlan created")
			body_unicode = re.content.decode('utf-8')
			content = json.loads(body_unicode)
			print(f"with experiment_id:{content['id']}\n")
			data = {"experiment_id":content['id']}
			#Send training request to server
			r = requests.post(f"{self.__url}training/", headers = header, data = data )
			body_unicode = r.content.decode('utf-8')
			content = json.loads(body_unicode)
			print(content['message'])


	def __getParameters(self):

		parameters = {'message': 'training',
					 'datasetId': self.__datasetId,
					 'epochs': self.__epochs,
					 'cycles': self.__cycles,
					 'modelName': self.__modelName,
					 'optimizer': self.__optimizer,
					 'lossFunction': self.__lossFunction,
					 'learningRate': self.__learningRate,
					 'stepsPerEpoch': self.__stepsPerEpoch,
					 'initialEpoch': self.__initialEpoch,
					 'validationSteps': self.__validationSteps,
					 'batchSize': self.__batchSize,
					 'featurewise_center': self.__featurewise_center,
					 'samplewise_center': self.__samplewise_center,
					 'featurewise_std_normalization': self.__featurewise_std_normalization,
					 'samplewise_std_normalization': self.__samplewise_std_normalization,
					 'zca_whitening': self.__zca_whitening,
					 'rotation_range': self.__rotation_range,
					 'width_shift_range': self.__width_shift_range,
					 'height_shift_range': self.__height_shift_range,
					 'brightness_range': self.__brightness_range,
					 'shear_range': self.__shear_range,
					 'zoom_range': self.__zoom_range,
					 'channel_shift_range': self.__channel_shift_range,
					 'fill_mode': self.__fill_mode,
					 'cval': self.__cval,
					 'horizontal_flip': self.__horizontal_flip,
					 'vertical_flip': self.__vertical_flip,
					 'rescale': self.__rescale,
					 'data_format': self.__data_format,
					 'validation_split': self.__validation_split,
					 'dtype': self.__dtype,
					 'shuffle': self.__shuffle,
					 'layersTrained': self.__layers_non_trainable,
					 'metrics': self.__metrics,
					 'objective': self.__objective,
					 'name': self.__name,
					 'modelType': self.__modelType,
					 'category': self.__category,
					 'upperboundTime': self.__upperboundTime
					 }

		return parameters

	def getTrainingPlan(self):

		print(f" \033[1mTraining Parameters\033[0m\n\n",
			f"objective: {self.__objective}\n",
			f"name: {self.__name}\n",
			f"modelType: {self.__modelType}\n",
			f"category: {self.__category}\n",
			f"datasetId: {self.__datasetId}\n",
			f"epochs: {self.__epochs}\n",
			f"cycles: {self.__cycles}\n",
			f"optimizer: {self.__optimizer}\n",
			f"lossFunction: {self.__lossFunction}\n",
			f"learningRate: {self.__learningRate}\n",
			f"stepsPerEpoch: {self.__stepsPerEpoch}\n",
			f"initialEpoch: {self.__initialEpoch}\n",
			f"validationSteps: {self.__validationSteps}\n",
			f"batchSize: {self.__batchSize}\n",
			f"metrics: {self.__metrics}\n",
			"\n \033[1mAugmentation Parameters\033[0m\n\n",
			f"featurewise_center: {self.__featurewise_center}\n",
			f"samplewise_center: {self.__samplewise_center}\n",
			f"featurewise_std_normalization': {self.__featurewise_std_normalization}\n",
			f"samplewise_std_normalization': {self.__samplewise_std_normalization}\n",
			f"zca_whitening': {self.__zca_whitening}\n",
			f"rotation_range': {self.__rotation_range}\n",
			f"width_shift_range': {self.__width_shift_range}\n",
			f"height_shift_range': {self.__height_shift_range}\n",
			f"brightness_range': {self.__brightness_range}\n",
			f"shear_range': {self.__shear_range}\n",
			f"zoom_range': {self.__zoom_range}\n",
			f"channel_shift_range': {self.__channel_shift_range}\n",
			f"fill_mode': {self.__fill_mode}\n",
			f"cval': {self.__cval}\n",
			f"horizontal_flip': {self.__horizontal_flip}\n",
			f"vertical_flip': {self.__vertical_flip}\n",
			f"rescale': {self.__rescale}\n",
			f"data_format': {self.__data_format}\n",
			f"validation_split': {self.__validation_split}\n",
			f"dtype': {self.__dtype}\n",
			f"shuffle': {self.__shuffle}\n",
			f"layersTrained': {self.__layers_non_trainable}\n")

		

