# SailGP Data Analyst Challenge

## Overview

This project is aimed at analyzing the data from the SailGP event in Auckland 2025. The data is provided in various formats and the goal is to answer specific questions using Python. The analysis includes data processing, visualization, and deriving insights from the provided datasets.

## Data Sources

- **Boat Logs**: Located in the `Boat_Logs` folder. The data is in CSV format and the columns are described in the `Boat_Logs/Boat_Logs_Columns.csv` file.
- **Course Marks**: The `Course_Marks_2025-01-19.csv` file contains the mark positions and wind readings on the course for the whole day.
- **Race XML**: The `Race_XML` folder contains XML files for each race that include information on course boundaries, theoretical mark positions, and target racecourse axis.
- **Manoeuvre Summary**: The `2025-01-19_man_summary.csv` file contains metrics from the manoeuvre summary for the day.
- **Straight Line Summary**: The `2025-01-19_straight_lines.csv` file contains metrics from the straight line summary for the day.
- **Polar Data**: The `2502 m8_APW_HSB2_HSRW.kph.csv` file contains the polar data for the boats in that configuration.

## Requirements

- Python 3.8 or higher
- Jupyter Notebook
- Suggested Libraries: `pandas`, `numpy`, `bokeh`, `beautifulsoup4`(to read xmls), `matplotlib`, `scipy`

## Setup

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required libraries:
    ```sh
    pip install pandas numpy bokeh matplotlib scipy etc....
    ```

3. Open the Jupyter Notebook:
    ```sh
    jupyter notebook main.ipynb
    ```

## Questions to Answer

1. **Calculate an accurate mean value for a compass direction (e.g., TWD or Heading) across a downsampled frequency.**
2. **Calculate a VMC column for a timeseries of boat Lat/Lon values using a course XML.**
3. **Verify and comment on the boats' calibration and propose a post-calibrated set of wind numbers.**
4. **Calculate a Distance to Leader metric for each boat using a timeseries of Lat/Lon positions and a course XML.**
5. **Calculate the minimum number of tacks or gybes for each leg of the course and each gate mark on the leg using a course XML, wind speed, direction, and polar data.**
6. **Calculate a “tacked” set of variables depending on the tack of the boat and visualize the results.**
7. **Train a model to explain the key features of tacks when optimizing for VMG and visualize the conclusions.**
8. **Provide insights on what made a team win or underperform in the race.**

## AI Assistance
You are welcome to use AI tools to assist you in your analysis, we all do now... 
If you do, sharing your preferred tools and how they helped you would be appreciated.

## Usage
- The notebook `main.ipynb` contains the code and analysis for the questions listed above.
- Each section of the notebook corresponds to a specific question and includes the necessary code, comments, and visualizations.


## Contact

For any questions or inquiries, please contact [drey@sailgp.com], [kkonig@sailgp.com].