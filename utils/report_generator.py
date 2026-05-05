def generate_area_report(district, poverty_index, built_up, vegetation, roads, nightlight):

    report = f"""
District: {district}

Numerical Socio-Economic Indicators
-----------------------------------
Poverty Index       : {poverty_index:.3f}
Built-up Score      : {built_up:.3f}
Vegetation Score   : {vegetation:.3f}
Road Density Score : {roads:.3f}
Nightlight Score   : {nightlight:.3f}

Interpretation:
Higher built-up, road, and nightlight scores generally indicate stronger infrastructure 
and economic activity, whereas higher vegetation with low infrastructure may indicate 
rural or underdeveloped regions. The poverty index reflects the combined effect of these indicators.
"""
    return report
