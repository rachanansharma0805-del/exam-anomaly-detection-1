# Dashboard Generation Script for Exam Anomaly Detection
# This script reads the anomaly_log.csv file, computes statistics and generates a visually appealing HTML dashboard summarizing the findings.
import pandas as pd
import os
from collections import Counter

# Anomaly_log path
possible_paths = [
    r"C:\Users\rachana sharma\exam-hall-anomaly\data\week3_anomaly_detection_finalised\anomaly_log.csv",
]

anomaly_log = None
csv_path = None

for path in possible_paths:
    if os.path.exists(path):
        csv_path = path
        print(f"✅ Found anomaly log: {path}")
        anomaly_log = pd.read_csv(path)
        break

if anomaly_log is None:
    print("ERROR: Cannot find anomaly_log.csv")
    print("\nPlease provide the correct path to your anomaly_log.csv file")
    print("Current directory:", os.getcwd())
    print("\nFiles in current directory:")
    for f in os.listdir('.'):
        print(f"  - {f}")
    exit(1)

# ==============================
# CALCULATE STATISTICS
# ==============================
total_anomalies = len(anomaly_log)
anomaly_types = Counter(anomaly_log['type'])
people_involved = anomaly_log['person_id'].nunique()

print(f"\n Processing {total_anomalies} anomalies...")

# ==============================
# GENERATE HTML DASHBOARD
# ==============================
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Exam Anomaly Detection Report</title>
    <meta charset="UTF-8">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 42px;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        .header p {{
            font-size: 18px;
            opacity: 0.9;
        }}
        .content {{
            padding: 40px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 25px;
            margin-bottom: 40px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        .stat-number {{
            font-size: 56px;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .stat-label {{
            color: #555;
            margin-top: 10px;
            font-size: 16px;
            font-weight: 600;
        }}
        .section {{
            background: #f8f9fa;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.05);
        }}
        .section h2 {{
            color: #333;
            font-size: 28px;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 3px solid #667eea;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            font-size: 14px;
            text-transform: uppercase;
        }}
        td {{
            padding: 15px;
            border-bottom: 1px solid #eee;
            color: #555;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        .anomaly-type {{
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .type-head {{ background: #ffebee; color: #c62828; }}
        .type-hand {{ background: #e3f2fd; color: #1565c0; }}
        .type-look {{ background: #fff3e0; color: #e65100; }}
        .type-raised {{ background: #f3e5f5; color: #6a1b9a; }}
        .type-proximity {{ background: #e8f5e9; color: #2e7d32; }}
        .insights {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .insights ul {{
            list-style: none;
            padding-left: 0;
        }}
        .insights li {{
            padding: 12px 0;
            border-bottom: 1px solid #eee;
            color: #555;
            font-size: 15px;
        }}
        .insights li:last-child {{
            border-bottom: none;
        }}
        .insights strong {{
            color: #667eea;
            font-weight: 600;
        }}
        .footer {{
            text-align: center;
            padding: 30px;
            background: #f8f9fa;
            color: #888;
            font-size: 14px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            background: #667eea;
            color: white;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1> Exam Anomaly Detection Report</h1>
            <p>Automated Surveillance System Analysis</p>
        </div>

        <div class="content">
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{total_anomalies}</div>
                    <div class="stat-label">Total Anomalies Detected</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{people_involved}</div>
                    <div class="stat-label">People Involved</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(anomaly_types)}</div>
                    <div class="stat-label">Anomaly Categories</div>
                </div>
            </div>

            <div class="section">
                <h2> Anomaly Distribution</h2>
                <table>
                    <tr>
                        <th>Anomaly Type</th>
                        <th>Count</th>
                        <th>Percentage</th>
                        <th>Severity</th>
                    </tr>
"""

for anom_type, count in anomaly_types.most_common():
    percentage = (count / total_anomalies) * 100
    css_class = "type-head" if "Head" in anom_type else \
                "type-hand" if "Hand" in anom_type else \
                "type-look" if "Looking" in anom_type else \
                "type-raised" if "Raised" in anom_type else "type-proximity"
    
    severity = "High" if percentage > 30 else "Medium" if percentage > 15 else "Low"
    
    html_content += f"""
                    <tr>
                        <td><span class="anomaly-type {css_class}">{anom_type}</span></td>
                        <td><strong>{count}</strong></td>
                        <td>{percentage:.1f}%</td>
                        <td><span class="badge">{severity}</span></td>
                    </tr>
    """

html_content += """
                </table>
            </div>

            <div class="section">
                <h2> Recent Detections</h2>
                <table>
                    <tr>
                        <th>Timestamp</th>
                        <th>Person ID</th>
                        <th>Anomaly Type</th>
                        <th>Frame</th>
                    </tr>
"""

display_count = min(20, len(anomaly_log))
for _, row in anomaly_log.head(display_count).iterrows():
    css_class = "type-head" if "Head" in row['type'] else \
                "type-hand" if "Hand" in row['type'] else \
                "type-look" if "Looking" in row['type'] else \
                "type-raised" if "Raised" in row['type'] else "type-proximity"
    
    html_content += f"""
                    <tr>
                        <td> {row['timestamp']}</td>
                        <td> #{row['person_id']}</td>
                        <td><span class="anomaly-type {css_class}">{row['type']}</span></td>
                        <td>#{row['frame']}</td>
                    </tr>
    """

if len(anomaly_log) > 20:
    html_content += f"""
                    <tr>
                        <td colspan="4" style="text-align: center; color: #888; padding: 20px;">
                            ... and <strong>{len(anomaly_log) - 20} more anomalies</strong> (view full log in CSV)
                        </td>
                    </tr>
    """

# Calculate insights
most_common = anomaly_types.most_common(1)[0]
most_active_person = anomaly_log['person_id'].mode()[0] if not anomaly_log.empty else "N/A"
person_anomaly_count = len(anomaly_log[anomaly_log['person_id'] == most_active_person])

# Calculate peak time
if not anomaly_log.empty and 'frame' in anomaly_log.columns:
    median_frame = anomaly_log['frame'].median()
    peak_time = f"{median_frame/30:.1f}s"
else:
    peak_time = "N/A"

html_content += f"""
                </table>
            </div>

            <div class="section">
                <h2>💡 Key Insights</h2>
                <div class="insights">
                    <ul>
                        <li> <strong>Most Common Anomaly:</strong> {most_common[0]} ({most_common[1]} occurrences, {most_common[1]/total_anomalies*100:.1f}% of total)</li>
                        <li> <strong>Person with Most Anomalies:</strong> #{most_active_person} ({person_anomaly_count} anomalies detected)</li>
                        <li> <strong>Peak Activity Time:</strong> Around {peak_time} into the exam</li>
                        <li> <strong>Average Anomalies per Person:</strong> {total_anomalies/people_involved:.1f}</li>
                        <li> <strong>Detection Rate:</strong> {total_anomalies/498*100:.1f}% of frames flagged (498 frames analyzed)</li>
                    </ul>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>Generated on {pd.Timestamp.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            <p>Exam Anomaly Detection System | Computer Vision & Deep Learning</p>
        </div>
    </div>
</body>
</html>
"""

# ==============================
# SAVE DASHBOARD
# ==============================
# Save in the same directory as the CSV
output_dir = os.path.dirname(csv_path) if csv_path else '.'
dashboard_path = os.path.join(output_dir, "anomaly_dashboard.html")

with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n Dashboard generated successfully!")
print(f"Location: {os.path.abspath(dashboard_path)}")
print(f"\n To view: Open the file in your web browser")
print(f"   Right-click → Open with → Chrome/Firefox/Edge")