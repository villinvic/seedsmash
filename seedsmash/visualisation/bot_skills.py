import plotly.graph_objects as go

# Data: Skill categories and values
categories = ['Edge Guarding', 'Neutral Game', 'Punish Game', 'Tech Chasing', 'Movement']
values_agent = [8.5, 7.0, 9.0, 6.5, 8.0]  # RL Agent's skills
values_human = [7.0, 8.0, 8.5, 7.5, 7.5]  # Human player's skills

# Close the loop for radar plot
categories += categories[:1]
values_agent += values_agent[:1]
values_human += values_human[:1]

# Create the radar plot
fig = go.Figure()

# RL Agent trace
fig.add_trace(go.Scatterpolar(
    r=values_agent,
    theta=categories,
    fill='toself',
    name='RL Agent',
    line_color='rgba(0, 123, 255, 0.8)',  # Blue
    fillcolor='rgba(0, 123, 255, 0.2)',  # Transparent blue fill
    line_width=2,
))

# Human trace
fig.add_trace(go.Scatterpolar(
    r=values_human,
    theta=categories,
    fill='toself',
    name='Human Player',
    line_color='rgba(255, 82, 82, 0.8)',  # Red
    fillcolor='rgba(255, 82, 82, 0.2)',  # Transparent red fill
    line_width=2,
))

# Customize the layout
fig.update_layout(
    title=dict(
        text='SSBM RL Agent Skill Comparison',
        font=dict(size=24, family='Arial, sans-serif', color='white'),
        x=0.5,  # Centered title
        xanchor='center'
    ),
    polar=dict(
        bgcolor='rgba(30, 30, 30, 1)',  # Dark background for the polar chart
        angularaxis=dict(
            tickfont=dict(size=14, family='Arial, sans-serif', color='white'),
            linecolor='rgba(0, 0, 0, 0)',  # Subtle gridline color
            gridcolor='rgba(0, 0, 0, 0)',  # Subtle angular gridlines
        ),
        radialaxis=dict(
            visible=True,
            tickfont=dict(size=12, color='white'),
            gridcolor='rgba(0, 0, 0, 0.)',  # Subtle radial gridlines
            linecolor='rgba(0, 0, 0, 0)',  # Subtle radial axis color
            range=[0, 10],  # Skill levels are between 0 and 10
        )
    ),
    showlegend=True,
    legend=dict(
        font=dict(size=14, color='white'),
        bgcolor='rgba(0, 0, 0, 0.5)',  # Transparent legend background
        bordercolor='rgba(255, 255, 255, 0.3)',
        borderwidth=1
    ),
    paper_bgcolor='rgba(20, 20, 20, 1)',  # Overall background color
    font=dict(color='white')  # Default font color
)

# Show the plot
fig.show()