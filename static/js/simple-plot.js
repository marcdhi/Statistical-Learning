// Simple script to test if Plotly is working correctly
document.addEventListener('DOMContentLoaded', function() {
    console.log("Simple Plotly test script loaded");
    
    // Get the plot container
    const plotContainer = document.getElementById('plot-area');
    
    if (!plotContainer) {
        console.error("Plot container not found");
        return;
    }
    
    // Only create the test plot if the container is empty
    if (plotContainer.innerHTML.trim() === '') {
        try {
            // Create a simple plot
            console.log("Creating a simple test plot");
            const data = [{
                x: [1, 2, 3, 4, 5],
                y: [1, 2, 4, 8, 16],
                type: 'scatter',
                mode: 'lines+markers',
                marker: {
                    color: '#1a365d',
                    size: 8
                },
                line: {
                    color: '#2c5282',
                    width: 2
                }
            }];
            
            const layout = {
                title: 'Simple Test Plot',
                xaxis: {
                    title: 'X Axis'
                },
                yaxis: {
                    title: 'Y Axis'
                }
            };
            
            // Create the plot
            Plotly.newPlot('plot-area', data, layout);
            console.log("Test plot created successfully");
        } catch (e) {
            console.error("Error creating test plot:", e);
            plotContainer.innerHTML = `<div class="text-center text-danger py-5">
                <p>Error creating test plot: ${e.message}</p>
            </div>`;
        }
    } else {
        console.log("Plot container already has content, skipping test plot");
    }
}); 