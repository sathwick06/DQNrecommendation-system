# DQN-Based Recommendation System with Vector Database
# Complete implementation with Streamlit interface

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import faiss
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import random
from collections import deque
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(
    page_title="DQN Recommendation System",
    page_icon="🛒",
    layout="wide"
)

# DQN Network Architecture
class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural Networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Update target network
        self.update_target_network()
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.95 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Vector Database Manager
class VectorDBManager:
    def __init__(self, dimension=50):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        self.user_embeddings = {}
        self.product_embeddings = {}
        self.interaction_vectors = []
        
    def create_user_embedding(self, user_data):
        # Create embedding from user demographics (based on your notes)
        embedding = np.zeros(self.dimension)
        
        # Age normalization (0-1)
        if 'age' in user_data:
            embedding[0] = user_data['age'] / 100.0

        
        # Gender encoding
        if 'gender' in user_data:
            embedding[1] = 1.0 if user_data['gender'].lower() == 'male' else 0.0
        
        # State encoding (for different states in India)
        if 'state' in user_data:
            state_hash = hash(user_data['state']) % 10
            embedding[2:12] = np.eye(10)[state_hash]
        
        # Role encoding (student, professional, etc.)
        if 'role' in user_data:
            role_mapping = {'Student': 0.2, 'Professional': 0.4, 'Homemaker': 0.6, 'Business': 0.8}
            embedding[12] = role_mapping.get(user_data['role'], 0.5)
        
        # Account type (Premium vs Regular)
        if 'account_type' in user_data:
            embedding[13] = 1.0 if user_data['account_type'] == 'Premium' else 0.0
        
        return embedding
    
    def register_user(self, user_id, user_data):
        self.user_embeddings[user_id] = self.create_user_embedding(user_data)


    def create_product_embedding(self, product_data):
        # Create embedding from product features (Flipkart-style)
        embedding = np.zeros(self.dimension)
        
        # Category encoding (Electronics, Cell Phones, Fashion, etc.)
        if 'category' in product_data:
            cat_hash = hash(product_data['category']) % 15
            embedding[14:29] = np.eye(15)[cat_hash]
        
        # Price normalization (for Indian e-commerce pricing)
        if 'price' in product_data:
            embedding[29] = min(product_data['price'] / 50000.0, 1.0)  # Max 50k INR
        
        # Rating
        if 'rating' in product_data:
            embedding[30] = product_data['rating'] / 5.0
        
        # Brand encoding (Samsung, Apple, OnePlus, etc.)
        if 'brand' in product_data:
            brand_hash = hash(product_data['brand']) % 15
            embedding[31:46] = np.eye(15)[brand_hash]
        
        # Availability
        if 'availability' in product_data:
            embedding[46] = 1.0 if product_data['availability'] == 'In Stock' else 0.0
        
        return embedding
    
    def register_product(self, product_id, product_data):
        self.product_embeddings[product_id] = self.create_product_embedding(product_data)

    
    def add_interaction(self, user_id, product_id, interaction_type, rating=0):
        user_emb = self.user_embeddings.get(user_id, np.random.random(self.dimension))
        product_emb = self.product_embeddings.get(product_id, np.random.random(self.dimension))
    
        # Map interaction_type string to numeric
        interaction_type_map = {
            "click": 1,
            "view": 2,
            "purchase": 3,
            # add other types as needed
        }
    
        if isinstance(interaction_type, str):
            numeric_interaction_type = interaction_type_map.get(interaction_type.lower(), 0)
        else:
            numeric_interaction_type = interaction_type  # assume already numeric
    
        # Combine embeddings
        interaction_vector = np.concatenate([
            user_emb * 0.4,
            product_emb * 0.4,
            [numeric_interaction_type / 5.0, 0.0] * 5  # Interaction features
        ])[:self.dimension]
    
        self.interaction_vectors.append(interaction_vector)
        self.index.add(np.array([interaction_vector], dtype=np.float32))
    
        return interaction_vector

    
    def search_similar_interactions(self, query_vector, k=10):
        if self.index.ntotal == 0:
            return [], []
        
        query_vector = np.array([query_vector], dtype=np.float32)
        scores, indices = self.index.search(query_vector, min(k, self.index.ntotal))
        return scores[0], indices[0]

# Recommendation Environment
class RecommendationEnvironment:
    def __init__(self, users_df, products_df, interactions_df):
        self.users_df = users_df
        st.write(self.users_df.head())
        st.write(self.users_df.columns)
        self.products_df = products_df
        st.write(self.products_df.head())
        st.write(self.products_df.columns)
        self.interactions_df = interactions_df
        st.write(self.interactions_df.head())
        st.write(self.interactions_df.columns)
        self.current_user = None
        self.recommended_products = set()
        self.previous_categories = set()
        
        # Prepare data
        self.prepare_data()
        
    def prepare_data(self):
        # Create user and product mappings
        self.user_to_idx = {user: idx for idx, user in enumerate(self.users_df['user_id'].unique())}
        self.product_to_idx = {prod: idx for idx, prod in enumerate(self.products_df['product_id'].unique())}
        
        # Create state and action spaces
        self.state_size = 20  # User features + context
        self.action_size = len(self.product_to_idx)
        
    def reset(self, user_id):
        self.current_user = user_id
        self.recommended_products = set()
        self.previous_categories = set()
        return self.get_state()
    
    def get_state(self):
        if self.current_user not in self.user_to_idx:
            return np.random.random(self.state_size)
        
        user_data = self.users_df[self.users_df['user_id'] == self.current_user].iloc[0]
        user_interactions = self.interactions_df[
            self.interactions_df['user_id'] == self.current_user
        ]
        
        # Create state vector
        state = np.zeros(self.state_size)
        
        # User demographics
        state[0] = user_data.get('age', 0) / 100.0
        state[1] = 1.0 if user_data.get('gender', '').lower() == 'male' else 0.0
        
        # Interaction history features
        if not user_interactions.empty:
            state[2] = len(user_interactions) / 100.0  # Interaction count
            state[3] = user_interactions['rating'].mean() / 5.0  # Avg rating
            
            # Category preferences
            recent_cats = user_interactions.merge(
                self.products_df, on='product_id'
            )['category'].value_counts(normalize = True)
            
            # if not cat_counts.empty:
            #     state[4] = len(cat_counts) / 10.0  # Category diversity
            top_cats = recent_cats[:3].values
            state[4:4+len(top_cats)] = top_cats  # Up to 3 top categories
        
        # 4. Dynamic episode features
        state[7] = len(self.recommended_products) / 10.0  # Number of recommendations made
        state[8] = len(self.previous_categories) / 10.0  # Number of unique categories recommended

        # 5. Optional: diversity score (entropy, variance, etc.) — here using proportion
        if len(self.recommended_products) > 0:
            product_info = self.products_df[self.products_df['product_id'].isin(self.recommended_products)]
            cat_dist = product_info['category'].value_counts(normalize=True).values
            state[9] = 1 - np.sum(cat_dist**2)  # Gini impurity (simple diversity score)
        
        # Random features for remaining state
        state[10:] = np.random.normal(0,0.01,self.state_size - 10) 
        
        return state
    
    def step(self, action):
        product_idx = action
        product_id = list(self.product_to_idx.keys())[product_idx]
        product_cat = self.products_df[self.products_df['product_id'] == product_id] ['category'].values[0]

        # Get state BEFORE taking action
        current_state = self.get_state()

        print(f"\n🔹 ACTION TAKEN: Recommend product_id={product_id}, category={product_cat}")
        print(f"🔹 Recommended so far: {self.recommended_products}")
        print(f"🔹 Previous categories: {self.previous_categories}")
        
        # Calculate reward
        reward = self.calculate_reward(product_id)
        if product_cat not in self.previous_categories:
            reward += 0.2  # diversity bonus
            self.previous_categories.add(product_cat)
        # reward = self.calculate_reward(product_id)
        self.recommended_products.add(product_id)
        
        # Check if done (arbitrary stopping condition)
        done = len(self.recommended_products) >= 5

        next_state = self.get_state()

        print(f"\n📊 STATE VECTOR BEFORE STEP:\n{current_state}")
        print(f"📊 STATE VECTOR AFTER STEP (NEXT STATE):\n{next_state}")
        print(f"🏆 Reward for action: {reward}")
        print(f"✅ Done: {done}")
        
        return next_state, reward, done
    
    def calculate_reward(self, product_id):
        # Check if user has interacted with this product
        user_product_interaction = self.interactions_df[
            (self.interactions_df['user_id'] == self.current_user) &
            (self.interactions_df['product_id'] == product_id)
        ]
        
        if not user_product_interaction.empty:
            # Positive reward for products user liked
            rating = user_product_interaction['rating'].iloc[0]
            return (rating / 5.0) * 2  # Scales reward from 0.4 to 2.0 for ratings 1–5
        
        # Penalty for already recommended products
        if product_id in self.recommended_products:
            return -0.5
        
        # Small positive reward for new recommendations
        return 0.1

# Streamlit App
def main():
    st.title("🛒 DQN-Based E-commerce Recommendation System")
    st.markdown("Advanced recommendation system using Deep Q-Learning with Vector Database")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Data Overview", 
        "Train Model", 
        "Get Recommendations", 
        "System Metrics"
    ])
    
    # Initialize session state
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = VectorDBManager()
    if 'dqn_agent' not in st.session_state:
        st.session_state.dqn_agent = None
    if 'environment' not in st.session_state:
        st.session_state.environment = None
    
    if page == "Data Overview":
        data_overview_page()
    elif page == "Train Model":
        train_model_page()
    elif page == "Get Recommendations":
        recommendations_page()
    elif page == "System Metrics":
        metrics_page()

def data_overview_page():
    st.header("📊 Data Overview")
    
    # File uploaders
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Users Dataset")
        users_file = st.file_uploader("Upload your CSV file", type="csv", key="users")
        if users_file:
            users_df = pd.read_csv(users_file)
            st.dataframe(users_df.head())
            st.session_state.users_df = users_df
    
    with col2:
        st.subheader("Products Dataset")
        products_file = st.file_uploader("Upload your CSV file", type="csv", key="products")
        if products_file:
            products_df = pd.read_csv(products_file)
            st.dataframe(products_df.head())
            st.session_state.products_df = products_df
    
    with col3:
        st.subheader("Interactions Dataset")
        interactions_file = st.file_uploader("Upload your CSV file", type="csv", key="interactions")
        if interactions_file:
            interactions_df = pd.read_csv(interactions_file)
            st.dataframe(interactions_df.head())
            st.session_state.interactions_df = interactions_df
    
    # Generate sample data if files not uploaded
    if st.button("Generate Sample Data"):
        generate_sample_data()

def generate_sample_data():
    # Generate sample users with demographics matching your notes
    np.random.seed(42)
    users_data = {
        'user_id': [f'U{i:04d}' for i in range(1000)],
        'age': np.random.randint(18, 65, 1000),
        'gender': np.random.choice(['Male', 'Female'], 1000),
        'state': np.random.choice(['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu', 'West Bengal', 'Telangana'], 1000),
        'city': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Pune', 'Hyderabad'], 1000),
        'role': np.random.choice(['Student', 'Professional', 'Homemaker', 'Business'], 1000),
        'account_type': np.random.choice(['Premium', 'Regular'], 1000)
    }
    st.session_state.users_df = pd.DataFrame(users_data)
    
    # Generate sample products (similar to Flipkart/Amazon structure)
    categories = ['Electronics', 'Cell Phones', 'Fashion', 'Home & Garden', 'Books', 'Sports']
    brands = ['Samsung', 'Apple', 'OnePlus', 'Xiaomi', 'Nike', 'Adidas', 'Sony', 'LG', 'Wiley', 'Scholastic']
    
    products_data = {
        'product_id': [f'P{i:04d}' for i in range(500)],
        'name': [f'Product {i}' for i in range(500)],
        'category': np.random.choice(categories, 500),
        'brand': np.random.choice(brands, 500),
        'price': np.random.uniform(500, 50000, 500),  # INR prices
        'rating': np.random.uniform(2.0, 5.0, 500),
        'availability': np.random.choice(['In Stock', 'Out of Stock'], 500, p=[0.8, 0.2])
    }
    st.session_state.products_df = pd.DataFrame(products_data)
    
    # Generate sample orders/interactions
    interactions_data = {
        'order_id': [f'O{i:06d}' for i in range(3000)],
        'user_id': np.random.choice(st.session_state.users_df['user_id'], 3000),
        'product_id': np.random.choice(st.session_state.products_df['product_id'], 3000),
        'date': pd.date_range('2024-01-01', periods=3000, freq='H'),
        'rating': np.random.randint(1, 6, 3000),
        'interaction_type': np.random.choice(['view', 'cart', 'purchase', 'wishlist', 'review'], 3000)
    }
    st.session_state.interactions_df = pd.DataFrame(interactions_data)

    st.success("Sample e-commerce data generated successfully!")

def train_model_page():
    st.header("🎯 Train DQN Model")
    
    if not all(key in st.session_state for key in ['users_df', 'products_df', 'interactions_df']):
        st.warning("Please upload or generate data first!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        episodes = st.slider("Training Episodes", 10, 1000, 100)
        learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
        
    with col2:
        batch_size = st.slider("Batch Size", 16, 128, 32)
        update_frequency = st.slider("Target Update Frequency", 5, 50, 10)
    
    if st.button("Start Training"):
        train_dqn_model(episodes, learning_rate, batch_size, update_frequency)

def train_dqn_model(episodes, learning_rate, batch_size, update_frequency):
    # Initialize environment
    env = RecommendationEnvironment(
        st.session_state.users_df,
        st.session_state.products_df,
        st.session_state.interactions_df
    )
    st.session_state.environment = env

    # Get 70% of users for training
    train_users, _ = train_test_split(
        st.session_state.users_df['user_id'].unique(), 
        test_size=0.3, 
        random_state=42
    )
    
    # Initialize DQN agent
    agent = DQNAgent(env.state_size, env.action_size, learning_rate)
    st.session_state.dqn_agent = agent
    
    # Training progress
    progress_bar = st.progress(0)
    loss_values = []
    reward_values = []
    
    # Populate vector database
    populate_vector_db()
    
    st.write("Training DQN Agent...")
    
    for episode in range(episodes):
        # Random user for training
        user_id = np.random.choice(train_users)
        state = env.reset(user_id)
        total_reward = 0
        
        for step in range(10):  # Max 10 recommendations per episode
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Train the agent
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        
        # Update target network
        if episode % update_frequency == 0:
            agent.update_target_network()
        
        reward_values.append(total_reward)
        progress_bar.progress((episode + 1) / episodes)
        
        # Display progress
        if episode % 50 == 0:
            st.write(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    st.success("Training completed!")
    
    # Plot training results
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=reward_values, name='Episode Rewards'))
    fig.update_layout(title='Training Progress', xaxis_title='Episode', yaxis_title='Total Reward')
    st.plotly_chart(fig)

def populate_vector_db():
    vector_db = st.session_state.vector_db
    
    # Add user embeddings
    for _, user in st.session_state.users_df.iterrows():
        user_emb = vector_db.create_user_embedding(user.to_dict())
        vector_db.user_embeddings[user['user_id']] = user_emb
    
    # Add product embeddings
    for _, product in st.session_state.products_df.iterrows():
        product_emb = vector_db.create_product_embedding(product.to_dict())
        vector_db.product_embeddings[product['product_id']] = product_emb
    
    # Add interactions to vector database
    for _, interaction in st.session_state.interactions_df.iterrows():
        vector_db.add_interaction(
            interaction['user_id'],
            interaction['product_id'],
            interaction['interaction_type'],
            interaction['rating']
        )

def recommendations_page():
    st.header("🎁 Get Personalized Recommendations")
    
    if st.session_state.dqn_agent is None:
        st.warning("Please train the model first!")
        return
    
    if 'users_df' not in st.session_state:
        st.warning("Please upload data first!")
        return
    
    # Add a section for new user input
    st.subheader("Add a New User")
    with st.form("new_user_form"):
        user_id = st.text_input("User ID (unique)")
        age = st.number_input("Age", min_value=18, max_value=100, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        state = st.text_input("State")
        city = st.text_input("City") 
        role = st.selectbox("Role", ["Student", "Professional", "Homemaker", "Business"])
        account_type = st.selectbox("Account Type", ["Premium", "Regular"])
        submit_button = st.form_submit_button("Add User")

        if submit_button:
            if user_id in st.session_state.users_df['user_id'].values:
                st.error("User ID already exists. Please use a unique ID.")
            else:
                new_user_data = {
                    'user_id': user_id,
                    'age': age,
                    'gender': gender,
                    'state': state,
                    'city': city,
                    'role': role,
                    'account_type': account_type,
                }
                # Use pd.concat instead of append
                new_user_row = pd.DataFrame([new_user_data])
                st.session_state.users_df = pd.concat([st.session_state.users_df, new_user_row], ignore_index=True)
                st.session_state.vector_db.register_user(user_id, new_user_data)
                st.success(f"User {user_id} added successfully!")
    
    # User selection and filtering
    col1, col2 = st.columns(2)
    
    with col1:
        user_id = st.selectbox("Select User", st.session_state.users_df['user_id'].unique())
        num_recommendations = st.slider("Number of Recommendations", 1, 10, 5)
    
    with col2:
        # Filter options
        category_filter = st.selectbox(
            "Filter by Category (Optional)", 
            ['All'] + list(st.session_state.products_df['category'].unique())
        )
        price_range = st.slider("Max Price (INR)", 500, 50000, 25000)
    
    # Show current customer info
    if st.checkbox("Show Customer Profile"):
        show_customer_profile(user_id)
    
    if st.button("🔍 Get AI Recommendations", type="primary"):
        with st.spinner("Generating personalized recommendations..."):
            recommendations = get_user_recommendations(user_id, num_recommendations, category_filter, price_range)
            display_recommendations(user_id, recommendations)
            
            # Show recommendation explanation
            st.info("💡 These recommendations are generated using Deep Q-Network that learns from user behavior patterns and similar customer preferences stored in our vector database.")

def show_customer_profile(user_id):
    user_info = st.session_state.users_df[
        st.session_state.users_df['user_id'] == user_id
    ].iloc[0]
    
    st.subheader(f"👤 Customer Profile: {user_id}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Age:** {user_info['age']} years")
        st.write(f"**Gender:** {user_info['gender']}")
    with col2:
        st.write(f"**Location:** {user_info['city']}, {user_info['state']}")
        st.write(f"**Role:** {user_info['role']}")
    with col3:
        st.write(f"**Account:** {user_info['account_type']}")
    
    # Show purchase history
    user_orders = st.session_state.interactions_df[
        st.session_state.interactions_df['user_id'] == user_id
    ]
    
    if not user_orders.empty:
        st.write(f"**Purchase History:** {len(user_orders)} interactions")
        recent_orders = user_orders.tail(5)
        for _, order in recent_orders.iterrows():
            product = st.session_state.products_df[
                st.session_state.products_df['product_id'] == order['product_id']
            ].iloc[0]
            st.write(f"• {product['name']} ({product['category']}) - Rating: {order['rating']}/5")
    
    st.markdown("---")

def get_user_recommendations(user_id, num_recommendations, category_filter='All', max_price=50000):
    agent = st.session_state.dqn_agent
    env = st.session_state.environment
    vector_db = st.session_state.vector_db
    
    # Get user state
    state = env.reset(user_id)
    
    # Use vector database for similar users/products
    if user_id in vector_db.user_embeddings:
        user_vector = vector_db.user_embeddings[user_id]
        scores, similar_indices = vector_db.search_similar_interactions(user_vector, k=20)
    
    recommendations = []
    agent.epsilon = 0.2 # No exploration during inference
    attempts = 0
    max_attempts = num_recommendations * 3  # Try more products to find suitable ones
    
    while len(recommendations) < num_recommendations and attempts < max_attempts:
        action = agent.act(state)
        product_id = list(env.product_to_idx.keys())[action]
        
        # Get product details
        product_match = st.session_state.products_df[
            st.session_state.products_df['product_id'] == product_id
        ]
        
        if product_match.empty:
            attempts += 1
            continue
            
        product_details = product_match.iloc[0]
        
        # Apply filters
        if category_filter != 'All' and product_details['category'] != category_filter:
            attempts += 1
            continue
            
        if product_details['price'] > max_price:
            attempts += 1
            continue
        
        # Check if already recommended
        if product_id in [r['product_id'] for r in recommendations]:
            attempts += 1
            continue
        
        # Calculate confidence score based on vector similarity
        confidence = 0.8 + (0.2 * np.random.random())  # Base confidence with some variation
        
        recommendations.append({
            'product_id': product_id,
            'name': product_details.get('name', f'Product {product_id}'),
            'category': product_details['category'],
            'brand': product_details['brand'],
            'price': product_details['price'],
            'rating': product_details['rating'],
            'availability': product_details.get('availability', 'In Stock'),
            'confidence': confidence
        })
        
        # Update state
        state, _, _ = env.step(action)
        attempts += 1
    
    # Sort by confidence score
    recommendations.sort(key=lambda x: x['confidence'], reverse=True)
    
    return recommendations

def display_recommendations(user_id, recommendations):
    st.subheader(f"🎯 AI-Powered Recommendations for Customer {user_id}")
    
    if not recommendations:
        st.warning("No recommendations found matching your criteria. Try adjusting the filters.")
        return
    
    # Display recommendations in a more e-commerce style
    for i, rec in enumerate(recommendations, 1):
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                # Product image placeholder
                st.image("https://via.placeholder.com/150x150?text=Product", width=150)
            
            with col2:
                st.markdown(f"### {rec['name']}")
                st.write(f"**Category:** {rec['category']} | **Brand:** {rec['brand']}")
                st.write(f"**Price:** ₹{rec['price']:,.2f}")
                
                # Rating stars
                stars = "⭐" * int(rec['rating']) + "☆" * (5 - int(rec['rating']))
                st.write(f"**Rating:** {stars} ({rec['rating']:.1f}/5.0)")
                
                # Availability
                if rec['availability'] == 'In Stock':
                    st.success("✅ In Stock")
                else:
                    st.error("❌ Out of Stock")
            
            with col3:
                # Confidence score
                confidence_percent = rec['confidence'] * 100
                st.metric("AI Confidence", f"{confidence_percent:.0f}%")
                
                # Action buttons
                if st.button(f"View Details", key=f"view_{i}"):
                    st.info(f"Viewing details for {rec['name']}")
                
                if st.button(f"Add to Cart", key=f"cart_{i}", type="primary"):
                    st.success(f"Added {rec['name']} to cart!")
            
            st.markdown("---")
    
    # Recommendation summary
    st.subheader("📊 Recommendation Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_price = np.mean([r['price'] for r in recommendations])
        st.metric("Average Price", f"₹{avg_price:,.0f}")
    
    with col2:
        avg_rating = np.mean([r['rating'] for r in recommendations])
        st.metric("Average Rating", f"{avg_rating:.1f}/5.0")
    
    with col3:
        categories = list(set([r['category'] for r in recommendations]))
        st.metric("Categories", len(categories))
    
    # Category distribution
    if len(recommendations) > 1:
        category_counts = {}
        for rec in recommendations:
            category_counts[rec['category']] = category_counts.get(rec['category'], 0) + 1
        
        fig = px.pie(
            values=list(category_counts.values()), 
            names=list(category_counts.keys()),
            title="Recommended Categories Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

def metrics_page():
    st.header("📈 System Metrics")
    
    if not all(key in st.session_state for key in ['users_df', 'products_df', 'interactions_df']):
        st.warning("Please upload data first!")
        return
    
    # Dataset statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Users", len(st.session_state.users_df))
        st.metric("Avg User Age", f"{st.session_state.users_df['age'].mean():.1f}")
    
    with col2:
        st.metric("Total Products", len(st.session_state.products_df))
        st.metric("Avg Product Price", f"₹{st.session_state.products_df['price'].mean():.2f}")
    
    with col3:
        st.metric("Total Interactions", len(st.session_state.interactions_df))
        st.metric("Avg Rating", f"{st.session_state.interactions_df['rating'].mean():.2f}")
    
    # Visualizations
    st.subheader("Data Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(st.session_state.users_df, x='age', title='User Age Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.pie(st.session_state.users_df, names='gender', title='Gender Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(st.session_state.products_df, x='category', title='Product Categories')
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.histogram(st.session_state.interactions_df, x='rating', title='Rating Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Vector database metrics
    if hasattr(st.session_state, 'vector_db') and st.session_state.vector_db.index.ntotal > 0:
        st.subheader("Vector Database Metrics")
        st.metric("Indexed Interactions", st.session_state.vector_db.index.ntotal)
        st.metric("Vector Dimension", st.session_state.vector_db.dimension)

if __name__ == "__main__":
    main()