import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import gradio as gr

css = '''
.gradio-container{max-width: 789px !important}
h1{text-align:center}
'''

# Function to create various visualizations from the data
def create_visualizations(data):
    plots = []
    
    # Create figures directory
    figures_dir = "./figures"
    shutil.rmtree(figures_dir, ignore_errors=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Histograms for numeric columns
    numeric_cols = data.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        plt.figure()
        sns.histplot(data[col], kde=True)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        hist_path = os.path.join(figures_dir, f'histogram_{col}.png')
        plt.savefig(hist_path)
        plt.close()
        plots.append(hist_path)
    
    # Box plots for numeric columns
    for col in numeric_cols:
        plt.figure()
        sns.boxplot(x=data[col])
        plt.title(f'Box Plot of {col}')
        box_path = os.path.join(figures_dir, f'boxplot_{col}.png')
        plt.savefig(box_path)
        plt.close()
        plots.append(box_path)
    
    # Scatter plot matrix for numeric columns
    if len(numeric_cols) > 1:
        plt.figure()
        sns.pairplot(data[numeric_cols])
        plt.title('Scatter Plot Matrix')
        scatter_matrix_path = os.path.join(figures_dir, 'scatter_matrix.png')
        plt.savefig(scatter_matrix_path)
        plt.close()
        plots.append(scatter_matrix_path)
    
    # Correlation heatmap for numeric columns
    if len(numeric_cols) > 1:
        plt.figure()
        corr = data[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        heatmap_path = os.path.join(figures_dir, 'correlation_heatmap.png')
        plt.savefig(heatmap_path)
        plt.close()
        plots.append(heatmap_path)
    
    # Bar charts for categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            plt.figure()
            data[col].value_counts().plot(kind='bar')
            plt.title(f'Bar Chart of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            bar_path = os.path.join(figures_dir, f'bar_chart_{col}.png')
            plt.savefig(bar_path)
            plt.close()
            plots.append(bar_path)
    
    # Line charts (if a 'date' column is present)
    if 'date' in data.columns:
        plt.figure()
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date').plot()
        plt.title('Line Chart of Date Series')
        line_chart_path = os.path.join(figures_dir, 'line_chart.png')
        plt.savefig(line_chart_path)
        plt.close()
        plots.append(line_chart_path)
    
    # Scatter plot using Plotly
    if len(numeric_cols) >= 2:
        fig = px.scatter(data, x=numeric_cols[0], y=numeric_cols[1], title='Scatter Plot')
        scatter_plot_path = os.path.join(figures_dir, 'scatter_plot.html')
        fig.write_html(scatter_plot_path)
        plots.append(scatter_plot_path)
    
    # Pie chart for categorical columns (only the first categorical column)
    if len(categorical_cols) > 0:
        fig = px.pie(data, names=categorical_cols[0], title='Pie Chart of ' + categorical_cols[0])
        pie_chart_path = os.path.join(figures_dir, 'pie_chart.html')
        fig.write_html(pie_chart_path)
        plots.append(pie_chart_path)
    
    # Heatmaps (e.g., for a correlation matrix or cross-tabulation)
    if len(numeric_cols) > 1:
        heatmap_data = data[numeric_cols].corr()
        fig = px.imshow(heatmap_data, text_auto=True, title='Heatmap of Numeric Variables')
        heatmap_plot_path = os.path.join(figures_dir, 'heatmap_plot.html')
        fig.write_html(heatmap_plot_path)
        plots.append(heatmap_plot_path)
    
    # Violin plots for numeric columns
    for col in numeric_cols:
        plt.figure()
        sns.violinplot(x=data[col])
        plt.title(f'Violin Plot of {col}')
        violin_path = os.path.join(figures_dir, f'violin_plot_{col}.png')
        plt.savefig(violin_path)
        plt.close()
        plots.append(violin_path)
    
    return plots

# Function to analyze the uploaded data
# Function to analyze the uploaded data with encoding error handling
def analyze_data(file_input):
    try:
        # Attempt to read the file using utf-8 encoding
        data = pd.read_csv(file_input.name, encoding='utf-8')
    except UnicodeDecodeError:
        # If utf-8 fails, try reading it with 'ISO-8859-1' encoding
        data = pd.read_csv(file_input.name, encoding='ISO-8859-1')
    
    # Generate visualizations
    visualizations = create_visualizations(data)
    return data, visualizations


# Define the Gradio Interface
with gr.Blocks(css=css) as demo:
    gr.Markdown("# DATA BOARD📊\nUpload a .csv file to generate various visualizations and interactive plots.")
    
    # File upload component
    file_input = gr.File(label="Upload your .csv file")
    
    # Submit button
    submit = gr.Button("Generate Dashboards")
    
    # Data table component to view the uploaded data
    data_table = gr.Dataframe(label="Data Table")
    
    # Gallery component to display visualizations
    gallery = gr.Gallery(label="Visualizations")
    
    # Submit action: analyze data and show in data table and gallery
    submit.click(analyze_data, inputs=file_input, outputs=[data_table, gallery])

if __name__ == "__main__":
    demo.launch(share=True)