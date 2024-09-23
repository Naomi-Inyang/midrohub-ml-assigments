from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import io
import base64

app = Flask(__name__)

# Load your dataset
df = pd.read_csv("Datasets/world_population.csv")
countries = df["Country/Territory"].unique()
df = df.iloc[:, [2] + list(range(5, 13))]

def create_country_population_dataframe(df: pd.DataFrame, country: str):
    country_data = df[df["Country/Territory"] == country]
    country_data.drop(columns=["Country/Territory"], inplace=True)

    new_df = pd.DataFrame()
    for col in country_data.columns:
        year = int(col.split()[0])
        population = country_data[col]
        new_row = pd.DataFrame({'Year': year, 'Population': population})
        new_df = pd.concat([new_df, new_row], ignore_index=True)
    
    new_df = new_df.iloc[::-1]
    new_df.set_index('Year', inplace=True)
    df.index = pd.to_datetime(df.index)

    return new_df

def forecast_population(selected_country):
    country_data = create_country_population_dataframe(df, selected_country)
    y = country_data.pop('Population')

    dp = DeterministicProcess(
        index=y.index,
        constant=True,
        order=2,
        drop=True,
    )
    X = dp.in_sample()

    lr_model = LinearRegression()
    lr_model.fit(X, y)
    y_fit = pd.Series(lr_model.predict(X), index=X.index)
    y_residual = y - y_fit

    xgb_model = XGBRegressor()
    xgb_model.fit(X, y_residual)
    y_fit_boosted = xgb_model.predict(X) + y_fit

    X_fore = dp.out_of_sample(steps=3, forecast_index=[2023, 2024, 2025])
    lr_fore = pd.Series(lr_model.predict(X_fore), index=X_fore.index)
    y_fore = xgb_model.predict(X_fore) + lr_fore 

    axs = y.plot(color='0.25', subplots=True, sharex=True)
    axs = y_fit_boosted.plot(color='C0', subplots=True, sharex=True, ax=axs)
    axs = y_fore.plot(color='C3', subplots=True, sharex=True, ax=axs)

    for ax in axs:
        ax.set_xlabel('Year')
        ax.set_ylabel('Population')
    for ax in axs:
        ax.legend([])

    _ = plt.suptitle(f"Population Forecast of {selected_country}")

    # Convert the plot to a PNG image in memory
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()

    # Encode the PNG image to base64
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    return img_base64

@app.route('/')
def index():
    return render_template('index.html', countries=countries)

@app.route('/', methods=['POST'])
def forecast():
    selected_country = request.form['country']
    image_data = forecast_population(selected_country)
    return render_template('index.html', countries=countries, image_data=image_data)

if __name__ == '__main__':
    app.run(debug=True)
