import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import scipy
import plotly.express as px
from datetime import datetime, timedelta
import io
import networkx as nx
from pyvis.network import Network
from streamlit.components.v1 import html

# Set page configuration
st.set_page_config(
    page_title="Dashboard PT XYZ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create session states
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""
if 'menu' not in st.session_state:
    st.session_state['menu'] = "Dashboard"
if 'page_number' not in st.session_state:
    st.session_state.page_number = 0
# Add date range state variables
if 'start_date' not in st.session_state:
    st.session_state['start_date'] = None
if 'end_date' not in st.session_state:
    st.session_state['end_date'] = None

# Custom CSS for styling including sidebar menu
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
<style>
    .login-header {
        margin-bottom: 2rem;
        text-align: center;
    }
    .login-form {
        width: 100%;
    }
    .main-header {
        color: #1E3A8A;
        padding-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB;
        margin-bottom: 2rem;
    }
    .welcome-text {
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .sidebar-header {
        font-weight: bold;
        margin-bottom: 1rem;
        padding: 1rem 0.5rem;
        border-bottom: 1px solid #e0e0e0;
    }
    .sidebar-button {
        display: flex;
        align-items: center;
        width: 100%;
        padding: 0.5rem 0.75rem;
        background-color: transparent;
        border: none;
        text-align: left;
        cursor: pointer;
        border-radius: 4px;
        margin: 0.25rem 0;
        transition: background-color 0.2s;
    }
    .sidebar-button:hover {
        background-color: #f5f5f5;
    }
    .sidebar-button.active {
        background-color: #e3f2fd;
        color: #1976d2;
    }
    .button-icon {
        font-size: 1.2rem;
        width: 1.5rem;
        margin-right: 0.75rem;
        text-align: center;
    }
    .button-text {
        font-size: 0.9rem;
        font-weight: 500;
    }
    .section-header {
        font-size: 0.8rem;
        color: #757575;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 1rem 0.75rem 0.5rem;
        margin-top: 1rem;
    }
    .divider {
        height: 1px;
        background-color: #e0e0e0;
        margin: 1rem 0;
    }
    .card {
        background-color: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    .card-header {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #1E3A8A;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Make Streamlit buttons look like our custom buttons */
    .stButton > button {
        display: flex;
        align-items: center;
        width: 100%;
        padding: 0.5rem 0.75rem;
        background-color: transparent;
        border: none;
        text-align: left;
        cursor: pointer;
        border-radius: 4px;
        margin: 0.25rem 0;
        transition: background-color 0.2s;
        color: #424242;
    }
    .stButton > button:hover {
        background-color: #f5f5f5;
        border: none;
    }
    .stButton > button:active {
        background-color: #e3f2fd;
        border: none;
        color: #1976d2;
    }
    /* Style for the date range warning */
    .date-warning {
        color: #ff4444;
        font-weight: bold;
        padding: 10px;
        background-color: #ffeeee;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    /* Style for the apply button */
    .apply-button {
        background-color: #1976d2 !important;
        color: white !important;
        font-weight: bold !important;
        padding: 0.5rem 1rem !important;
        width: 100% !important;
        text-align: center !important;
        display: block !important;
        margin-top: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# Dummy user data 
valid_users = {
    "admin": "admin123",
    "analyst": "analyst123",
    "manager": "manager123",
    "executive": "executive123"
}

# Login function
def login(username, password):
    if username in valid_users and valid_users[username] == password:
        st.session_state['logged_in'] = True
        st.session_state['username'] = username
        return True
    return False

# Logout function
def logout():
    st.session_state['logged_in'] = False
    st.session_state['username'] = ""
    st.session_state['menu'] = "Dashboard"

# Display login page
def show_login():
    
    st.markdown('<div class="login-header">', unsafe_allow_html=True)
    
    st.markdown('<h2 style="text-align: center;">Dashboard Social Network Analysis PT XYZ</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; margin-bottom: 2rem;">Please login to access the dashboard</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="login-form">', unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if login(username, password):
            st.success("Login successful! Redirecting...")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Invalid username or password")
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; color: #6B7280; font-size: 0.8rem;">
        © 2025 PT XYZ. All rights reserved.<br>
        For support, please contact: support@ptxyz.com
    </div>
    """, unsafe_allow_html=True)

# Function to create a sidebar menu item with icon
def sidebar_menu_item(icon, label, menu_id):
    is_active = st.session_state.menu == menu_id
    return f"""
    <div class="button-icon">
        <i class="{icon}"></i>
    </div>
    <div class="button-text">{label}</div>
    """

# Load data function
@st.cache_data
def load_data():
    try:
        # Update this path to your actual dataset path - for demo purposes, we'll create a synthetic dataset
        return pd.read_excel("data/all-data.xlsx")
        
        # Create a synthetic dataset for demonstration
        date_range = pd.date_range(start='2023-01-01', end='2025-05-01', freq='D')
        np.random.seed(42)
        
        # Generate random counts with some patterns
        counts = np.random.poisson(lam=10, size=len(date_range))
        # Add some seasonality (higher on weekends)
        counts[date_range.dayofweek >= 5] *= 1.5
        # Add some trend (increasing over time)
        counts = counts + np.linspace(0, 5, len(date_range))
        
        # For network data
        usernames = [f"user_{i}" for i in range(1, 101)]
        
        # Create main dataframe
        df = pd.DataFrame({
            'tanggal-upload': date_range,
            'id': range(1, len(date_range) + 1),
            'source': np.random.choice(['facebook', 'twitter', 'instagram'], size=len(date_range)),
            'engagement': np.random.randint(10, 1000, size=len(date_range)),
            'username': np.random.choice(usernames, size=len(date_range)),
            'in_reply_to_screen_name': np.random.choice(usernames + [None], size=len(date_range))
        })
        
        return df
        
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        # Return empty dataframe if data can't be loaded
        return pd.DataFrame()

# Plot time series function - IMPROVED
def plot_series(df, date_col='tanggal-upload', start_date=None, end_date=None, series_name='', series_index=0):
    try:
        palette = list(sns.color_palette("tab10"))
        
        # Make a copy to avoid modification warnings
        df_copy = df.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        
        # Filter by date range if provided
        if start_date and end_date:
            mask = (df_copy[date_col] >= start_date) & (df_copy[date_col] <= end_date)
            filtered_df = df_copy[mask]
        else:
            filtered_df = df_copy
        
        # Check if we have data
        if filtered_df.empty:
            return None, None, "No data available for the selected date range."
        
        # Count occurrences and prepare data
        counted = (filtered_df[date_col]
                    .value_counts()
                    .reset_index(name='counts')
                    .rename({'index': date_col}, axis=1)
                    .sort_values(date_col, ascending=True))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
        ax.plot(counted[date_col], counted['counts'], label=series_name, 
                color=palette[series_index % len(palette)], marker='o', markersize=4)
        
        # Format date axis for better readability
        plt.gcf().autofmt_xdate()
        
        # Add labels and styling
        sns.despine(fig=fig, ax=ax)
        plt.xlabel('Upload Date', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Number of Uploads by Date', fontsize=14, fontweight='bold')
        
        # Add grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        return fig, counted, None
    except Exception as e:
        return None, None, f"Error creating plot: {str(e)}"


# Dashboard content function - IMPROVED
def show_dashboard():
    st.markdown('<h2><i class="fas fa-chart-line"></i> Dashboard</h2>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-text">Welcome to the Social Network Analysis Dashboard. This dashboard provides insights into network relationships and patterns within PT XYZ.</p>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.warning("Dataset is not available. Please check the dataset path.")
        return
    
    # Ensure date column is in datetime format
    if 'tanggal-upload' in df.columns:
        df['tanggal-upload'] = pd.to_datetime(df['tanggal-upload'], errors='coerce')
    else:
        st.warning("Required column 'tanggal-upload' not found in dataset.")
        return
    
    # Get min and max dates from data
    min_date = df['tanggal-upload'].min().date()
    max_date = df['tanggal-upload'].max().date()
    
    # Date range selection in the sidebar
    st.sidebar.markdown('### Date Range Selection')
    
    # Add preset date ranges
    preset_ranges = st.sidebar.selectbox(
        "Preset Date Ranges",
        ["Custom", "Last 7 Days", "Last 30 Days", "Last 90 Days", "Last Year", "All Time"],
        index=5  # Default to All Time
    )
    
    # Set date range based on preset
    data_end_date = max_date
    if preset_ranges == "Last 7 Days":
        default_start = data_end_date - timedelta(days=7)
        default_end = data_end_date
    elif preset_ranges == "Last 30 Days":
        default_start = data_end_date - timedelta(days=30)
        default_end = data_end_date
    elif preset_ranges == "Last 90 Days":
        default_start = data_end_date - timedelta(days=90)
        default_end = data_end_date
    elif preset_ranges == "Last Year":
        default_start = data_end_date - timedelta(days=365)
        default_end = data_end_date
    elif preset_ranges == "All Time":
        default_start = min_date
        default_end = max_date
    else:  # Custom
        # Use previously selected dates if available, otherwise defaults
        default_start = st.session_state.get('start_date', min_date)
        default_end = st.session_state.get('end_date', max_date)
    
    # Allow manual date selection with proper bounds
    st.sidebar.markdown("##### Select Date Range")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date", 
            value=default_start, 
            min_value=min_date, 
            max_value=max_date
        )
    
    with col2:
        # Ensure end_date can't be before start_date
        end_date = st.date_input(
            "End Date", 
            value=default_end if default_end >= start_date else start_date,
            min_value=start_date,  # This ensures end_date can't be before start_date
            max_value=max_date
        )
    
    # Save selected dates to session state
    st.session_state['start_date'] = start_date
    st.session_state['end_date'] = end_date
    
    # Apply button for updating the visualization
    if st.sidebar.button("Apply Date Range", key="apply_date_range", help="Click to apply the selected date range"):
        st.rerun()
    
    # Convert to datetime for filtering
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    
    # Show data statistics
    st.sidebar.markdown('### Data Statistics')
    data_in_range = df[(df['tanggal-upload'] >= start_datetime) & (df['tanggal-upload'] <= end_datetime)]
    
    if data_in_range.empty:
        st.sidebar.warning("No data in selected date range")
    else:
        st.sidebar.write(f"Total entries in selected range: {len(data_in_range)}")
    
    # Overview metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="card-header"><i class="fas fa-calendar"></i> Date Range</div>', unsafe_allow_html=True)
        st.write(f"**From:** {start_date}")
        st.write(f"**To:** {end_date}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card-header"><i class="fas fa-upload"></i> Total Uploads</div>', unsafe_allow_html=True)
        st.write(f"**Count:** {len(data_in_range)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="card-header"><i class="fas fa-chart-line"></i> Daily Average</div>', unsafe_allow_html=True)
        days_diff = max(1, (end_date - start_date).days + 1)
        daily_avg = len(data_in_range) / days_diff
        st.write(f"**Average:** {daily_avg:.2f} uploads/day")
        st.markdown('</div>', unsafe_allow_html=True)
    
  
    # Social Network Analysis section
    st.subheader("Network characteristics", divider="grey")
    
    # Creating a graph from dataframe
    with st.spinner('Creating social network...'):
        try:
            # Filter data to selected date range
            filtered_data = df[(df['tanggal-upload'] >= start_datetime) & 
                            (df['tanggal-upload'] <= end_datetime)]
            
            # Remove rows where in_reply_to_screen_name is None
            network_data = filtered_data.dropna(subset=['in_reply_to_screen_name'])
            
            if network_data.empty:
                st.warning("No network data available for the selected date range.")
            else:
                # Create graph
                my_graph = nx.from_pandas_edgelist(
                    network_data, 
                    source="username", 
                    target="in_reply_to_screen_name"
                )
                my_graph.remove_edges_from(nx.selfloop_edges(my_graph))
                
                # Calculate metrics
                num_nodes = my_graph.number_of_nodes()
                num_edges = my_graph.number_of_edges()
                density = nx.density(my_graph)
                
                # Calculate connected components
                connected_components = list(nx.connected_components(my_graph))
                num_components = len(connected_components)
                
                # Display metrics in columns
                col1, col2, col3 = st.columns(3)
                with col1:

                    st.metric("Node", num_nodes)
                    st.metric('Connected Components', num_components)
                with col2:
                    st.metric("Edges",num_edges)

                    if nx.is_connected(my_graph):
                        avg_path = nx.average_shortest_path_length(my_graph)
                        st.metric("Average Shortest Path",f"{avg_path:.3f}")
                    else:
                        if connected_components:
                            largest_cc = max(connected_components, key=len)
                            subgraph = my_graph.subgraph(largest_cc)
                            avg_path = nx.average_shortest_path_length(subgraph)
                            st.metric("Average Shortest Path (Largest Component)", f"{avg_path:.3f}")
                with col3:
                    st.metric("Network Density",f"{density:.4f}")
            
                # Network visualization with Pyvis
                st.markdown("### Network Visualization")
                
        except Exception as e:
            st.error(f"Error in network analysis: {str(e)}")
    
    vis_tab1, vis_tab2 = st.tabs(["Visualisasi SNA", "Visualisasi DNA"])
    with vis_tab1:
        try:
            net = Network(height="600px", width="100%", bgcolor="#FFFFFF", font_color="#333333")
            
            # Add nodes and edges
            for node in my_graph.nodes():
                net.add_node(node, label=node)
            
            for edge in my_graph.edges():
                net.add_edge(edge[0], edge[1])
            
            # Set physics and options
            net.toggle_physics(True)
            net.set_options("""
            const options = {
                "nodes": {
                    "borderWidth": 2,
                    "size": 30,
                    "color": {
                        "border": "#222222",
                        "background": "#3498db"
                    },
                    "font": {"size": 12}
                },
                "edges": {
                    "color": "#757575",
                    "smooth": false
                },
                "physics": {
                    "forceAtlas2Based": {
                        "gravitationalConstant": -50,
                        "centralGravity": 0.01,
                        "springLength": 100,
                        "springConstant": 0.08
                    },
                    "maxVelocity": 50,
                    "solver": "forceAtlas2Based",
                    "timestep": 0.35,
                    "stabilization": {"iterations": 150}
                }
            }
            """)
            
            # Save and display the network
            try:
                path = "network.html"
                net.save_graph(path)
                with open(path, 'r', encoding='utf-8') as f:
                    html_string = f.read()
                
                # Display network
                html(html_string, height=600)
            except Exception as e:
                st.error(f"Error generating network visualization: {str(e)}")
                
    
            col1, col2, col3 = st.columns(3)
            
            # Kolom 1: Degree Centrality
            with col1:
                st.subheader("Degree Centrality")
                # Mengambil top 10 node berdasarkan degree
                degree_data = sorted(nx.degree(my_graph), key=lambda x: x[1], reverse=True)[0:10]
                
                # Mengkonversi ke dataframe
                degree_df = pd.DataFrame(degree_data, columns=['Node', 'Degree'])
                
                # Menampilkan tabel
                st.dataframe(degree_df)
                
                # Menambahkan deskripsi
                #st.write("Degree centrality mengukur jumlah koneksi langsung yang dimiliki oleh setiap node.")
            
            # Kolom 2: Closeness Centrality
            with col2:
                st.subheader("Closeness Centrality")
                # Mengambil top 10 node berdasarkan closeness centrality
                closeness_data = sorted(nx.closeness_centrality(my_graph).items(), 
                                    key=lambda x: x[1], reverse=True)[:10]
                
                # Mengkonversi ke dataframe
                closeness_df = pd.DataFrame(closeness_data, columns=['Node', 'Closeness'])
                
                # Menampilkan tabel
                st.dataframe(closeness_df)
                
                # Menambahkan deskripsi
                #st.write("Closeness centrality mengukur seberapa dekat node dengan semua node lain dalam jaringan.")
            
            # Kolom 3: Betweenness Centrality
            with col3:
                st.subheader("Betweenness Centrality")
                # Mengambil top 10 node berdasarkan betweenness centrality
                betweenness_data = sorted(nx.betweenness_centrality(my_graph, 
                                                                normalized=True, 
                                                                endpoints=True).items(),
                                        key=lambda x: x[1], reverse=True)[:10]
                
                # Mengkonversi ke dataframe
                betweenness_df = pd.DataFrame(betweenness_data, columns=['Node', 'Betweenness'])
                
                # Menampilkan tabel
                st.dataframe(betweenness_df)
                
                # Menambahkan deskripsi
                #st.write("Betweenness centrality mengukur seberapa sering node berada pada jalur terpendek antara node lain dalam jaringan.")
        
        except Exception as e:
            st.error(f"Error in network analysis: {str(e)}")    
    
    # Tab 2: Visualisasi dengan Plotly
    with vis_tab2:
        try:
            st.markdown('<div class="card-header"><i class="fas fa-chart-area"></i> Upload Activity Over Time</div>', unsafe_allow_html=True)
            
            # Create and display plot with improved error handling
            fig, counted_data, error_msg = plot_series(
                df, 
                date_col='tanggal-upload', 
                start_date=start_datetime, 
                end_date=end_datetime
            )
            
            if error_msg:
                st.warning(error_msg)
            elif fig:
                st.pyplot(fig)
                
            else:
                st.warning("No data available for the selected date range.")
            
            st.markdown('</div>', unsafe_allow_html=True)
              
            col1, col2 = st.columns(2)
            with col1:
                st.image("images/DNA-Q1.png", caption="Dynamic Network Analysis Quarter 1")
                st.image("images/DNA-Q3.png", caption="Dynamic Network Analysis Quarter 3") 
                # Display data table
                st.subheader('Daily Upload Counts')
                
                if counted_data is not None and not counted_data.empty:
                    # Make sure index is reset
                    daily_counts = counted_data.reset_index(drop=True)
                    daily_counts = daily_counts.rename(columns={'tanggal-upload': 'Date', 'counts': 'Count'})
                    
                    # Format date for better readability
                    daily_counts['Date'] = daily_counts['Date'].dt.strftime('%Y-%m-%d')

                    st.dataframe(daily_counts, width=800, height=210)
                
                else:
                    st.warning("No data available for the selected date range.")
                
                # Close the div tag properly
                st.markdown('</div>', unsafe_allow_html=True)


            with col2:
                st.image("images/DNA-Q2.png", caption="Dynamic Network Analysis Quarter 2")
                st.image("images/DNA-Q4.png", caption="Dynamic Network Analysis Quarter 4") 
                quarterly_data = {'Pembagian Data (Quarter)': [
                            'Quarter 1 (Q1)',
                            'Quarter 2 (Q2)', 
                            'Quarter 3 (Q3)',
                            'Quarter 4 (Q4)',
                            'Full Year'
                        ],
                        'Periode Tanggal': [
                            'Januari - Maret 2024',
                            'April - Juni 2024',
                            'Juli - September 2024', 
                            'Oktober - Desember 2024',
                            'Januari - Desember 2024'
                        ] }
                # Create DataFrame
                quarterly_df = pd.DataFrame(quarterly_data)

                # Display the table in Streamlit
                st.subheader("Pembagian Data per Quarter")
                st.dataframe(quarterly_df, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)

            #qquarter 1 
            st.subheader("**Centrality Quarter 1 (Q1)**", divider="blue" ) 
                
            dc1, dc2, dc3= st.columns(3) 
            with dc1: 
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.subheader("Degree Centrality") 
                dc_data_q1 = { 'Node': [
                    'convomf',
                    'tanyakanrl',
                    'tanyarlfes',
                    'Mahalinidisini',
                    'starfess',
                    'initiaraandini',
                    'ssefnum',
                    'eudelienta',
                    'tinkerabels',
                    'utokki_u'
                ], 
                'Degree': [
                    111,
                    62,
                    27,
                    20,
                    16,
                    16,
                    15,
                    14,
                    13,
                    12
                ] 
            }
                dc_data_q1 = pd.DataFrame(dc_data_q1)
                st.dataframe( dc_data_q1,use_container_width=True, hide_index=False)
            with dc2:
                st.markdown('</div>', unsafe_allow_html=True)
                st.subheader("Betwennes Centrality") 
                # Hasil konversi dalam format dictionary
                bc_data_q1 = {  
                    'Node': [
                        'convomf',
                        'tanyakanrl',
                        'saputelbang',
                        'tanyarlfes',
                        'initiaraandini',
                        'glassesofTiara',
                        'catuwagapat',
                        'bobocuuu',
                        'starfess',
                        'timutttt'
                    ], 
                    'Degree': [
                        0.011488764784910831,
                        0.0059110007639419405,
                        0.0026124399603101403,
                        0.0021739416769842733,
                        0.0015471470719949424,
                        0.0014905099971022893,
                        0.0014863390497264716,
                        0.0014863390497264716,
                        0.00136999157029583,
                        0.0013248794815731936
                    ] 
                }
                bc_data_q1 = pd.DataFrame(bc_data_q1)
                st.dataframe(bc_data_q1, use_container_width=True, hide_index=False)
            with dc3: 
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.subheader("Closeness Centrality")
                cc_data_q1 = {'Node': [
                        'convomf',
                        'saputelbang',
                        'catuwagapat',
                        'bobocuuu',
                        'tanyakanrl',
                        'miwamikomiyu',
                        'iammpuding',
                        'glassesofTiara',
                        'timutttt',
                        'stcarlordd'
                    ],
                    'Closeness Centrality': [
                        0.05152633767150038,
                        0.045270070205247266,
                        0.04237926112445242,
                        0.04237926112445242,
                        0.03814133501200718,
                        0.03775080939754977,
                        0.03775080939754977,
                        0.03724238098815516,
                        0.03707593459267737,
                        0.03658540403577315
                    ]
                }

                # Create DataFrame
                cc_data_q1 = pd.DataFrame(cc_data_q1)
                st.dataframe(cc_data_q1, use_container_width=True, hide_index=False)


            #QUARTER 2
            st.subheader("Centrality Quarter 2 (Q2)", divider="blue" ) 
            
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
               
                st.markdown('</div>', unsafe_allow_html=True) 
                st.subheader("Degree Centrality")
                dc_data_q2 = {  
                    'Node': [
                        'convomf',
                        'tanyakanrl',
                        'kegblgnunfaedh',
                        'tanyarlfes',
                        'starfess',
                        'MUSIKMENFESS',
                        'Mahalinidisini',
                        'samsungID',
                        'ssefnum',
                        'initiaraandini'
                    ], 
                    'Degree': [
                        421,
                        239,
                        102,
                        100,
                        82,
                        42,
                        39,
                        35,
                        33,
                        29
                    ] 
                }               
                dc_data_q2 = pd.DataFrame(dc_data_q2)
                st.dataframe( dc_data_q2,use_container_width=True, hide_index=False)

            with bc2:    
                st.markdown('</div>', unsafe_allow_html=True)
                st.subheader("Betwennes Centrality") 
                bc_data_q2 = {  
                    'Node': [
                        'convomf',
                        'tanyakanrl',
                        'tanyarlfes',
                        'officialinews_',
                        'kegblgnunfaedh',
                        'starfess',
                        'SINDOnews',
                        'UJunami',
                        'tinkerabels',
                        'initiaraandini'
                    ], 
                    'Degree': [
                        0.031442252224304336,
                        0.021518143631214893,
                        0.014991713021802355,
                        0.011173253143177391,
                        0.00734265923676037,
                        0.006428932045869842,
                        0.005192261858497747,
                        0.004299897842017884,
                        0.0042317080197576224,
                        0.004071769379106158
                    ] 
                }
                bc_data_q2 = pd.DataFrame(bc_data_q2)
                st.dataframe(bc_data_q2, use_container_width=True, hide_index=False)
            with bc3:    
                st.markdown('</div>', unsafe_allow_html=True)
                st.subheader("Closeness Centrality")
                
                cc_data_q2 = {'Node': [
                        'officialinews_',
                        'convomf',
                        'SINDOnews',
                        'tanyakanrl',
                        'saputelbang',
                        'tanyarlfes',
                        'bobocuuu',
                        'iammpuding',
                        'needumdbstr',
                        'Nayadeul'
                    ],
                    'Closeness Centrality': [
                        0.08157411277484973,
                        0.07891261325164703,
                        0.07682126137303773,
                        0.07294035631327804,
                        0.07294035631327804,
                        0.06788121111314363,
                        0.06687698044503795,
                        0.06674767005922404,
                        0.06661885876612728,
                        0.06643238190464058
                    ]
                }

                cc_data_q2 = pd.DataFrame(cc_data_q2)
                st.dataframe(cc_data_q2, use_container_width=True, hide_index=False)

            #QUARTER 3
            
            st.subheader("**Centrality Quarter 3 (Q3)**", divider="blue" ) 
            ac1, ac2, ac3= st.columns(3)
            with ac1: 
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.subheader("Degree Centrality")
                dc_data_q3 = {  
                    'Node': [
                        'convomf',
                        'tanyakanrl',
                        'tanyarlfes',
                        'kegblgnunfaedh',
                        'starfess',
                        'kabarheboh01',
                        'MUSIKMENFESS',
                        'Mahalinidisini',
                        'SakuraRelapse',
                        'initiaraandini'
                    ], 
                    'Degree': [
                        475,
                        269,
                        150,
                        115,
                        104,
                        75,
                        56,
                        44,
                        39,
                        35
                    ] 
                }
                dc_data_q3 = pd.DataFrame(dc_data_q3)
                st.dataframe( dc_data_q3,use_container_width=True, hide_index=False)

            with ac2:
                st.markdown('</div>', unsafe_allow_html=True)
                st.subheader("Betwenness Centrality")
                # Hasil konversi dalam format dictionary
                bc_data_q3 = {  
                    'Node': [
                        'convomf',
                        'tanyakanrl',
                        'officialinews_',
                        'tanyarlfes',
                        'starfess',
                        'SINDOnews',
                        'kegblgnunfaedh',
                        'kabarheboh01',
                        'storyfeyy',
                        'initiaraandini'
                    ], 
                    'Degree': [
                        0.028643469573382936,
                        0.01995435693445714,
                        0.011751785970417214,
                        0.009023686826459208,
                        0.00784131526204695,
                        0.006063128633088182,
                        0.00578095799726552,
                        0.004005166957407937,
                        0.003948346161537709,
                        0.003364039528934205
                    ] 
                }
                bc_data_q3 = pd.DataFrame(bc_data_q3)
                st.dataframe(bc_data_q3, use_container_width=True, hide_index=False)
            with ac3: 
                st.markdown('</div>', unsafe_allow_html=True)
                st.subheader("Closeness Centrality")
                
                cc_data_q3 = {'Node': [
                        'convomf',
                        'officialinews_',
                        'tanyakanrl',
                        'SINDOnews',
                        'storyfeyy',
                        'saputelbang',
                        'okezonenews',
                        'tanyarlfes',
                        'needumdbstr',
                        'bobocuuu'
                    ],
                    'Closeness Centrality': [
                        0.07395134676278854,
                        0.07395134676278854,
                        0.0703179718452477,
                        0.06927255598839346,
                        0.06577988386287739,
                        0.06543952332335458,
                        0.06495935910609726,
                        0.06376296798767034,
                        0.06321014747778131,
                        0.06314573933892072
                    ]
                }

                # Create DataFrame
                cc_data_q3 = pd.DataFrame(cc_data_q3)
                st.dataframe(cc_data_q3, use_container_width=True, hide_index=False)



            #QUARTER 4
            
            st.subheader("Centrality Quarter 4 (Q4)", divider="blue") 
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.subheader("Degree Centrality")
                dc_data_q4 = {  
                    'Node': [
                        'convomf',
                        'tanyakanrl',
                        'tanyarlfes',
                        'kegblgnunfaedh',
                        'starfess',
                        'MUSIKMENFESS',
                        'kabarheboh01',
                        'sbyfess',
                        'IndoPopBase',
                        'Mahalinidisini'
                    ], 
                    'Degree': [
                        510,
                        304,
                        166,
                        126,
                        120,
                        85,
                        75,
                        49,
                        46,
                        44
                    ] 
                }
                dc_data_q4 = pd.DataFrame(dc_data_q4)
                st.dataframe( dc_data_q4,use_container_width=True, hide_index=False)      
            with cc2:
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.subheader("Betweenness Centrality ")
                bc_data_q4 = {  
                    'Node': [
                        'convomf',
                        'tanyakanrl',
                        'officialinews_',
                        'tanyarlfes',
                        'starfess',
                        'storyfeyy',
                        'kegblgnunfaedh',
                        'SINDOnews',
                        'MUSIKMENFESS',
                        'initiaraandini'
                    ], 
                    'Degree': [
                        0.030820891335969002,
                        0.01980199162733408,
                        0.010958795465014734,
                        0.008970796978080819,
                        0.008268614145050987,
                        0.006900513033213817,
                        0.006512149765768965,
                        0.005561385492255017,
                        0.005119174267672106,
                        0.0036847589690573003
                    ] 
                }
                bc_data_q4 = pd.DataFrame(bc_data_q4)
                st.dataframe(bc_data_q4, use_container_width=True, hide_index=False)
            
            with cc3: 
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.subheader("Closeness Centrality")
                cc_data_q4 = {'Node': [
                        'convomf',
                        'officialinews_',
                        'tanyakanrl',
                        'SINDOnews',
                        'storyfeyy',
                        'saputelbang',
                        'okezonenews',
                        'tanyarlfes',
                        'zaynkhalifah',
                        'panggilajaayu'
                    ],
                    'Closeness Centrality': [
                        0.0773021966628912,
                        0.07430989694864394,
                        0.0723586799357624,
                        0.0703853155904763,
                        0.06844536297745459,
                        0.066666659538131,
                        0.0661401808084245,
                        0.06536087033830228,
                        0.06510185759195326,
                        0.06500772964931213
                    ]
                }
                cc_data_q4=pd.DataFrame(cc_data_q4)
                st.dataframe(cc_data_q4, use_container_width=True, hide_index=False)

        except Exception as e:
            st.error(f"Error in network analysis: {str(e)}")
    st.markdown("---")
    st.write(f"Total nodes: {my_graph.number_of_nodes()}, Total edges: {my_graph.number_of_edges()}")  

# Dataset section function
def show_dataset():
    st.markdown('<h2><i class="fas fa-database"></i> Dataset</h2>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-text">Explore and analyze the datasets used for social network analysis.</p>', unsafe_allow_html=True)
        
        # Load data
    df = load_data()
        
    if not df.empty:
        # Display dataset information
        st.markdown('<h3>Dataset Information</h3>', unsafe_allow_html=True)
            
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")
        st.markdown('</div>', unsafe_allow_html=True)
            
            # Pagination for data
        st.markdown('<h3>Dataset Preview</h3>', unsafe_allow_html=True)
        rows_per_page = 12
        total_pages = (len(df) + rows_per_page - 1) // rows_per_page
            
            # Display data for selected page
        start_idx = st.session_state.page_number * rows_per_page
        end_idx = min(start_idx + rows_per_page, len(df))
            
        st.write(f"Showing rows {start_idx} to {end_idx} of {len(df)}")
        st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True, height=500)

        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("Previous", disabled=(st.session_state.page_number <= 0)):
                st.session_state.page_number -= 1
                st.rerun()
                    
        with col2:
            page_number = st.slider("Page", 0, max(0, total_pages-1), st.session_state.page_number)
            st.session_state.page_number = page_number
                
        with col3:
            if st.button("Next", disabled=(st.session_state.page_number >= total_pages-1)):
                st.session_state.page_number += 1
                st.rerun()
            
        st.markdown('</div>', unsafe_allow_html=True)
            
            # Display column information
        st.markdown('<h3>Column Details</h3>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isna().sum()
            }), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
            
            # Store the dataset in session state for use in other sections
        st.session_state['Dataset'] = df
    else:
        st.error("Error loading dataset. Please check if the dataset file is available in the specified path.")

# FAQ section function
def show_faq():
    st.markdown('<h2><i class="fas fa-question-circle"></i> Frequently Asked Questions</h2>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-text">Find answers to common questions about social network analysis and how to interpret the results.</p>', unsafe_allow_html=True)
    
    st.markdown("<h3>How to use the date range selection?</h3>", unsafe_allow_html=True)
    st.write("""
    1. Go to the Dashboard page
    2. In the sidebar, select a preset date range or choose "Custom"
    3. If using custom, set your desired Start Date and End Date
    4. Click the "Apply Date Range" button to update visualizations
    5. The charts and data tables will update to reflect your selected date range
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>What if no data appears for my selected date range?</h3>", unsafe_allow_html=True)
    st.write("""
    If you see a warning message that no data is available for your selected date range, try the following:
    
    1. Check that your start date is before your end date
    2. Try expanding your date range to include more data
    3. Check the Dataset page to understand the available date ranges in your data
    4. Use the "All Time" preset to see all available data
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>How can I download the visualization or data?</h3>", unsafe_allow_html=True)
    st.write("""
    On the Dashboard page:
    
    1. After selecting your desired date range, locate the chart area
    2. Below the chart, click the "Download Chart" button to save the chart as a PNG image
    3. In the Daily Upload Counts section, click "Download Data as CSV" to export the data
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>What do the network metrics mean?</h3>", unsafe_allow_html=True)
    st.write("""
    - **Nodes:** Represents individual entities (users) in the network
    - **Edges:** Connections between nodes (interactions between users)
    - **Network Density:** Measures how interconnected the network is (ranges from 0 to 1, where 1 is fully connected)
    - **Connected Components:** Groups of nodes that are connected to each other but disconnected from other groups
    - **Average Shortest Path:** The average number of steps needed to reach one node from another
    """)

def main():
    # Sidebar configuration
    with st.sidebar:
        st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
        #st.image("https://via.placeholder.com/150x80?text=PT+XYZ", width=150)
        st.markdown(f"<p>Welcome, <b>{st.session_state['username'].capitalize()}</b></p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Main navigation using styled buttons
        col1, col2 = st.columns([1, 5])
        with col1:
            st.markdown('<i class="fas fa-chart-line" style="font-size: 1.8rem;"></i>', unsafe_allow_html=True)
        with col2:
            if st.button("Dashboard", key="dashboard", use_container_width=True):
                st.session_state.menu = "Dashboard"
                st.rerun()
        
        col1, col2 = st.columns([1, 5])
        with col1:
            st.markdown('<i class="fas fa-database" style="font-size: 1.8rem;"></i>', unsafe_allow_html=True)
        with col2:
            if st.button("Dataset", key="dataset", use_container_width=True):
                st.session_state.menu = "Dataset"
                st.rerun()
        
        col1, col2 = st.columns([1, 5])
        with col1:
            st.markdown('<i class="fas fa-question-circle" style="font-size: 1.8rem;"></i>', unsafe_allow_html=True)
        with col2:
            if st.button("FAQ", key="faq", use_container_width=True):
                st.session_state.menu = "FAQ"
                st.rerun()
        
        
        # Logout button
        col1, col2 = st.columns([1, 5])
        with col1:
            st.markdown('<i class="fas fa-sign-out-alt" style="font-size: 1.8rem;"></i>', unsafe_allow_html=True)
        with col2:
            if st.button("Logout", key="logout", use_container_width=True):
                logout()
                st.rerun()
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Main content
    st.markdown(f'<h1 class="main-header">Social Network Analysis - {st.session_state.menu}</h1>', unsafe_allow_html=True)
    
    # Display content based on selected menu
    if st.session_state.menu == "Dashboard":
        show_dashboard()
    elif st.session_state.menu == "Dataset":
        show_dataset()
    elif st.session_state.menu == "FAQ":
        show_faq()


# Application flow
if not st.session_state['logged_in']:
    show_login()
else:
    main()