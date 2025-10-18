# AFFRITECH ML COURSE: COMPLETE TABLE OF CONTENTS
## From Zero to AI Hero - A Project-Based Journey

---

## **COURSE OVERVIEW**

**Total Episodes:** 14 Core + 4 Bonus = 18 Episodes
**Duration:** ~2 hours per episode
**Prerequisites:** None - Complete beginner friendly
**Tools:** Google Colab, PyTorch, Python
**Outcome:** Portfolio of 18+ ML projects

---

## **PHASE 1: FOUNDATIONS (Episodes 1-4)**
### *Building Your ML Foundation*

---

### **Episode 1: Your First AI in 60 Minutes** ✅ COMPLETED
**Concepts:**
- Google Colab setup
- Python basics (variables, data types, operations)
- Lists and loops
- Functions
- Simple pattern recognition
- Linear regression from scratch
- Least squares method

**Project:** Number Pattern Predictor
- Input-output pattern detection
- Manual implementation of linear regression
- Making predictions on new data

**Skills Unlocked:**
- Python fundamentals
- Basic ML concepts
- Pattern recognition
- Mathematical thinking

---

### **Episode 2: Teaching Machines Like Teaching Dogs**
**Concepts:**
- Introduction to scikit-learn
- Training vs. testing data
- Train-test split
- Model evaluation metrics (MSE, R², MAE)
- Overfitting vs. underfitting
- Feature scaling/normalization
- Data visualization with matplotlib

**Project:** House Price Predictor
- Real dataset with multiple features
- Linear regression using scikit-learn
- Feature importance analysis
- Visualization of predictions vs. actual
- Error analysis
- Model performance comparison

**Skills Unlocked:**
- Using ML libraries
- Data splitting
- Model evaluation
- Data visualization
- Understanding model performance

**Homework:**
Build a predictor for: car prices, student grades, or salary estimation

---

### **Episode 3: Data Cleaning - The Unsexy Truth**
**Concepts:**
- Pandas for data manipulation
- Handling missing data (dropna, fillna, interpolation)
- Detecting and removing outliers
- Data types and conversion
- Encoding categorical variables (one-hot, label encoding)
- Feature engineering basics
- Data exploration (describe, info, value_counts)
- Correlation analysis

**Project:** Real Estate Data Cleanup & Analysis
- Load messy real-world dataset
- Handle missing values intelligently
- Remove outliers
- Create new features (price per sqft, age of property)
- Encode categorical variables (neighborhood, property type)
- Build cleaned predictor model
- Compare: dirty data vs. clean data performance

**Skills Unlocked:**
- Data cleaning pipeline
- Pandas mastery
- Feature engineering
- Data quality assessment

**Homework:**
Clean and analyze a Kaggle dataset of your choice

---

### **Episode 4: Multiple Features, Multiple Powers**
**Concepts:**
- Multiple linear regression
- Feature selection techniques
- Polynomial regression
- Regularization (Ridge, Lasso)
- Cross-validation
- Hyperparameter tuning
- Model comparison
- Bias-variance tradeoff

**Project:** Advanced House Price Predictor
- Multiple features (size, bedrooms, location, age, etc.)
- Polynomial features for non-linear relationships
- Feature selection (remove unimportant features)
- Regularization to prevent overfitting
- K-fold cross-validation
- Hyperparameter tuning with GridSearchCV
- Final model comparison and selection

**Skills Unlocked:**
- Advanced regression techniques
- Model selection
- Preventing overfitting
- Systematic model improvement

**Homework:**
Build a multi-feature predictor for: laptop prices, flight delays, or energy consumption

---

## **PHASE 2: CLASSIFICATION & DECISION MAKING (Episodes 5-7)**
### *Teaching Machines to Make Decisions*

---

### **Episode 5: Yes or No? Classification Basics**
**Concepts:**
- Classification vs. regression
- Logistic regression
- Decision boundaries
- Classification metrics (accuracy, precision, recall, F1-score)
- Confusion matrix
- ROC curve and AUC
- Class imbalance handling

**Project:** Email Spam Detector
- Binary classification (spam vs. not spam)
- Text feature extraction (bag of words, TF-IDF)
- Logistic regression model
- Evaluate with multiple metrics
- Confusion matrix visualization
- Threshold tuning for precision/recall tradeoff
- Test on real email examples

**Skills Unlocked:**
- Classification fundamentals
- Text processing basics
- Model evaluation for classification
- Handling imbalanced data

**Homework:**
Build a classifier for: fake news detection, sentiment analysis (positive/negative reviews), or fraud detection

---

### **Episode 6: Decision Trees - How Machines Think**
**Concepts:**
- Decision tree algorithm
- Information gain and entropy
- Tree depth and pruning
- Random forests (ensemble learning)
- Feature importance from trees
- Bagging and bootstrap aggregation
- Out-of-bag error

**Project:** Medical Diagnosis Assistant
- Multi-class classification (healthy, flu, cold, COVID)
- Symptoms as features
- Decision tree visualization
- Random forest for improved accuracy
- Feature importance (which symptoms matter most?)
- Interpretable predictions
- Confidence scores

**Skills Unlocked:**
- Tree-based models
- Ensemble methods
- Model interpretability
- Feature importance analysis

**Homework:**
Build a decision system for: loan approval, customer segmentation, or plant disease classification

---

### **Episode 7: Support Vector Machines - Drawing the Best Line**
**Concepts:**
- SVM intuition and math
- Kernel trick (linear, polynomial, RBF)
- Margin maximization
- Support vectors
- Multi-class SVM
- SVM for non-linear problems
- Hyperparameter tuning (C, gamma)

**Project:** Handwritten Digit Recognizer (MNIST Subset)
- Image classification with SVM
- Pixel values as features
- Kernel selection and comparison
- Hyperparameter optimization
- Visualization of decision boundaries
- Misclassification analysis
- Real-time drawing prediction

**Skills Unlocked:**
- SVM theory and practice
- Kernel methods
- Image data handling
- Advanced classification

**Homework:**
Build an SVM classifier for: iris flower species, wine quality classification, or face recognition (simple dataset)

---

## **PHASE 3: INTRODUCTION TO NEURAL NETWORKS (Episodes 8-10)**
### *Building Brains from Scratch*

---

### **Episode 8: Neural Networks - The Brain Connection**
**Concepts:**
- Biological neuron vs. artificial neuron
- Perceptron model
- Activation functions (sigmoid, tanh, ReLU)
- Forward propagation
- Loss functions
- Backpropagation intuition
- Gradient descent
- Learning rate
- Epochs and batch size

**Project:** Build a Neural Network from Scratch
- Implement a 2-layer network in pure Python/NumPy
- XOR problem solution
- Visualize activation functions
- Watch gradient descent in action
- Training animation
- Compare with scikit-learn's MLPClassifier
- Understand what libraries do under the hood

**Skills Unlocked:**
- Neural network fundamentals
- Backpropagation understanding
- Optimization basics
- Low-level implementation

**Homework:**
Extend the network to 3 layers and solve a more complex problem (circle classification)

---

### **Episode 9: PyTorch - Your Neural Network Playground**
**Concepts:**
- Introduction to PyTorch
- Tensors and operations
- Automatic differentiation (autograd)
- Building models with nn.Module
- Loss functions in PyTorch
- Optimizers (SGD, Adam)
- Training loops
- Model saving and loading
- GPU acceleration basics

**Project:** Fashion Item Classifier (Fashion-MNIST)
- Build a fully connected neural network
- Classify 10 types of clothing
- Data loading with DataLoader
- Training with mini-batches
- Validation and early stopping
- Learning curves visualization
- Test on uploaded images
- Model checkpointing

**Skills Unlocked:**
- PyTorch fundamentals
- Deep learning workflow
- Model training best practices
- Data pipeline creation

**Homework:**
Build a neural network for: CIFAR-10 subset, sign language digits, or audio classification

---

### **Episode 10: Convolutional Neural Networks - Computer Vision Unlocked**
**Concepts:**
- Why CNNs for images?
- Convolutional layers
- Filters and feature maps
- Pooling layers (max, average)
- Padding and stride
- CNN architectures
- Transfer learning introduction
- Image augmentation

**Project:** Dog Breed Classifier
- Build a CNN from scratch
- Use transfer learning (ResNet, VGG)
- Fine-tuning pre-trained models
- Data augmentation (rotation, flip, zoom)
- Classify 10+ dog breeds
- Grad-CAM visualization (what does the model see?)
- Deploy as a simple web interface
- Test on user-uploaded photos

**Skills Unlocked:**
- CNN architecture design
- Transfer learning
- Data augmentation
- Model visualization
- Practical deployment

**Homework:**
Build a CNN for: plant disease detection, food classification, or traffic sign recognition

---

## **PHASE 4: SEQUENCE MODELS & NLP (Episodes 11-12)**
### *Teaching Machines to Understand Language and Time*

---

### **Episode 11: Recurrent Neural Networks - Memory Matters**
**Concepts:**
- Sequential data and time series
- Vanilla RNN architecture
- Vanishing gradient problem
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Bidirectional RNNs
- Sequence-to-sequence models
- Teacher forcing

**Project:** Text Generation Bot
- Character-level or word-level RNN
- Train on Shakespeare, code, or tweets
- Temperature-based sampling
- Generate creative text
- Autocomplete system
- Story continuation
- Compare LSTM vs. GRU performance
- Interactive generation interface

**Skills Unlocked:**
- RNN/LSTM architecture
- Sequence modeling
- Text generation
- Sampling strategies

**Homework:**
Build a sequence model for: stock price prediction, music generation, or recipe generator

---

### **Episode 12: Natural Language Processing - Talking to Machines**
**Concepts:**
- Tokenization and text preprocessing
- Word embeddings (Word2Vec, GloVe)
- Embedding layers
- Sentiment analysis
- Text classification
- Attention mechanism introduction
- Transformer basics (high-level)
- Pre-trained models (BERT overview)

**Project:** Movie Review Sentiment Analyzer
- Binary sentiment classification
- Text preprocessing pipeline
- Word embeddings visualization (t-SNE)
- LSTM-based classifier
- Attention weights visualization
- Real-time sentiment analysis
- Test on tweets, product reviews
- Emoji prediction based on text
- Build a simple API

**Skills Unlocked:**
- NLP fundamentals
- Text classification
- Embeddings and attention
- Practical NLP applications

**Homework:**
Build an NLP system for: toxic comment detection, question classification, or news category prediction

---

## **PHASE 5: GENERATIVE AI & ADVANCED TOPICS (Episodes 13-14)**
### *Creating, Not Just Predicting*

---

### **Episode 13: Autoencoders - Compression and Reconstruction**
**Concepts:**
- Autoencoder architecture
- Encoder-decoder structure
- Latent space representation
- Denoising autoencoders
- Variational autoencoders (VAE)
- Anomaly detection with autoencoders
- Dimensionality reduction
- Image reconstruction

**Project:** Image Denoiser & Anomaly Detector
- Build an autoencoder
- Remove noise from images
- Compress images to latent space
- Reconstruct images
- Detect anomalies in manufacturing (defect detection)
- Generate new samples with VAE
- Latent space interpolation
- Face generation experiments

**Skills Unlocked:**
- Autoencoder design
- Unsupervised learning
- Anomaly detection
- Generative modeling basics

**Homework:**
Build an autoencoder for: audio denoising, credit card fraud detection, or handwriting style transfer

---

### **Episode 14: Generative Adversarial Networks - Creating Reality**
**Concepts:**
- GAN architecture (generator vs. discriminator)
- Adversarial training
- Mode collapse
- Training stability tricks
- DCGAN (Deep Convolutional GAN)
- Conditional GANs
- StyleGAN overview
- Ethical considerations

**Project:** Face Generator & Style Transfer
- Build a simple GAN
- Generate synthetic faces
- Use pre-trained StyleGAN
- Face interpolation (morphing)
- Attribute editing (add glasses, change age)
- Image-to-image translation
- Deepfake awareness
- Ethical discussion on synthetic media

**Skills Unlocked:**
- GAN training
- Generative modeling
- Image synthesis
- AI ethics awareness

**Homework:**
Experiment with GANs for: anime character generation, landscape creation, or fashion design

---

## **BONUS EPISODES**

---

### **Bonus 1: Reinforcement Learning - Training Game Players**
**Concepts:**
- RL fundamentals (agent, environment, reward)
- Q-learning
- Deep Q-Networks (DQN)
- Policy gradients
- Actor-critic methods
- Exploration vs. exploitation
- OpenAI Gym

**Project:** Train an AI to Play CartPole & Atari Games
- Implement Q-learning
- Build a DQN
- Train agent to balance CartPole
- Play simple Atari games
- Visualize learning progress
- Compare different RL algorithms
- Watch agent improve over time

**Skills Unlocked:**
- Reinforcement learning basics
- Agent training
- Reward engineering
- Game AI

---

### **Bonus 2: Model Deployment - From Notebook to Production**
**Concepts:**
- Model serving
- Flask/FastAPI basics
- Docker containerization
- REST APIs
- Streamlit for quick UIs
- Gradio for ML demos
- Cloud deployment (Hugging Face Spaces, Heroku)
- Model optimization (quantization, pruning)
- Monitoring and logging

**Project:** Deploy Your Best Model as a Web App
- Choose your best project
- Create a REST API
- Build a web interface (Streamlit/Gradio)
- Containerize with Docker
- Deploy to cloud
- Share publicly
- Add to portfolio
- Monitor usage

**Skills Unlocked:**
- Model deployment
- API development
- Web development basics
- Cloud platforms
- Portfolio building

---

### **Bonus 3: Computer Vision Advanced - Object Detection & Segmentation**
**Concepts:**
- Object detection (YOLO, R-CNN)
- Bounding boxes
- Semantic segmentation
- Instance segmentation
- Transfer learning for detection
- Real-time video processing
- Multi-object tracking

**Project:** Real-Time Object Detector
- Implement YOLO
- Detect multiple objects in images
- Real-time webcam detection
- Count objects in video
- Build a people counter
- Vehicle detection system
- Custom object detector training

**Skills Unlocked:**
- Object detection
- Real-time processing
- Video analysis
- Advanced computer vision

---

### **Bonus 4: Time Series & Forecasting - Predicting the Future**
**Concepts:**
- Time series analysis
- Stationarity and trends
- ARIMA models
- Prophet for forecasting
- LSTM for time series
- Multi-variate forecasting
- Seasonality and trends
- Evaluation metrics for forecasting

**Project:** Stock Price & Weather Forecaster
- Analyze time series data
- Build ARIMA models
- LSTM-based forecasting
- Prophet for business metrics
- Multiple forecasting horizons
- Confidence intervals
- Anomaly detection in time series
- Interactive forecast dashboard

**Skills Unlocked:**
- Time series analysis
- Forecasting techniques
- Business analytics
- Temporal modeling

---

## **CAPSTONE PROJECT (Final 2 Weeks)**

### **Build Your Dream ML Application**

**Choose One Track:**

**Track 1: Computer Vision Application**
- Medical image diagnosis
- Wildlife monitoring system
- Autonomous vehicle component
- AR filter creator
- Quality control system

**Track 2: NLP Application**
- Chatbot with personality
- Document summarizer
- Code generator
- Language learning assistant
- Content moderation system

**Track 3: Time Series/Forecasting**
- Stock trading bot
- Energy consumption optimizer
- Demand forecasting system
- Predictive maintenance
- Climate analysis tool

**Track 4: Multi-Modal Application**
- Image captioning system
- Visual question answering
- Video summarizer
- Audio-visual classifier
- Multi-sensor fusion system

**Requirements:**
- End-to-end ML pipeline
- Deployed and accessible
- Documentation
- Presentation/demo
- GitHub repository
- Blog post or video explanation

---

## **SKILLS PROGRESSION MAP**

### **After Episode 4:**
- ✅ Python programming
- ✅ Data manipulation
- ✅ Classical ML algorithms
- ✅ Model evaluation
- ✅ Data cleaning
- **Can build:** Prediction systems, data analysis tools

### **After Episode 7:**
- ✅ Classification algorithms
- ✅ Tree-based models
- ✅ SVM and kernel methods
- ✅ Model comparison
- **Can build:** Decision systems, classifiers, diagnostic tools

### **After Episode 10:**
- ✅ Neural networks
- ✅ Deep learning with PyTorch
- ✅ CNNs for computer vision
- ✅ Transfer learning
- **Can build:** Image classifiers, visual recognition systems

### **After Episode 12:**
- ✅ RNNs and LSTMs
- ✅ NLP fundamentals
- ✅ Sequence modeling
- ✅ Text processing
- **Can build:** Text analyzers, chatbots, sequence predictors

### **After Episode 14:**
- ✅ Generative models
- ✅ Autoencoders
- ✅ GANs
- ✅ Advanced architectures
- **Can build:** Creative AI, synthesis systems, anomaly detectors

### **After Bonus Episodes:**
- ✅ Reinforcement learning
- ✅ Model deployment
- ✅ Advanced computer vision
- ✅ Time series forecasting
- **Can build:** Production ML systems, complete applications

---

## **COURSE OUTCOMES**

### **Portfolio Projects:**
- 14+ complete ML projects
- 1 capstone project
- All code on GitHub
- Deployed applications
- Blog posts/documentation

### **Technical Skills:**
- Python (NumPy, Pandas, Matplotlib)
- Scikit-learn
- PyTorch
- Computer Vision
- NLP
- Deployment
- Data Science workflow

### **Career Readiness:**
- Portfolio website
- GitHub profile
- Technical blog
- Interview preparation
- Project presentations
- Real-world problem solving

---

## **RECOMMENDED LEARNING PATH**

### **If You're Interested in Computer Vision:**
Episodes 1-4 → 8-10 → Bonus 3 → Capstone (Track 1)

### **If You're Interested in NLP:**
Episodes 1-4 → 8-9 → 11-12 → Capstone (Track 2)

### **If You're Interested in Data Science:**
Episodes 1-7 → Bonus 4 → Capstone (Track 3)

### **If You Want Everything:**
Follow the sequential order → All bonuses → Capstone (Track 4)

---

## **TIME COMMITMENT**

- **Per Episode:** 2 hours class + 2-3 hours homework = ~5 hours/week
- **Total Course:** 14 episodes × 5 hours = 70 hours
- **Bonus Content:** 4 episodes × 5 hours = 20 hours
- **Capstone:** 10-15 hours
- **Total:** ~100-105 hours (3-4 months at 1-2 episodes/week)

---

## **PREREQUISITES BY PHASE**

### **Phase 1 (Episodes 1-4):**
- None! Complete beginner friendly

### **Phase 2 (Episodes 5-7):**
- Phase 1 completion
- Comfortable with Python
- Understand basic ML workflow

### **Phase 3 (Episodes 8-10):**
- Phases 1-2 completion
- Strong Python skills
- Understand calculus basics (derivatives)
- Linear algebra basics (matrices, vectors)

### **Phase 4 (Episodes 11-12):**
- Phases 1-3 completion
- Neural network understanding
- PyTorch proficiency

### **Phase 5 (Episodes 13-14):**
- All previous phases
- Deep learning experience
- Comfortable with advanced concepts

### **Bonus Episodes:**
- Can be taken after relevant core episodes
- Independent of each other

---

**End of Table of Contents**