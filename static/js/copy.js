  
  (async function() {
    const data = [{
      date: '2003',
      value: 1.12,
  },
  {
      date: '2004',
      value: 1.35,
  },
  {
      date: '2005',
      value: 3.22,
  },
  {
      date: '2006',
      value: 4.96,
  },
  {
      date: '2007',
      value: 5.02,
  },
  {
      date: '2008',
      value: 1.92,
  },
  {
      date: '2009',
      value: 0.15,
  },
  {
      date: '2010',
      value: 0.17,
  },
  {
      date: '2011',
      value: 0.10,
  },
  {
      date: '2012',
      value: 0.14,
  },
  {
      date: '2013',
      value: 0.10,
  },
  {
      date: '2014',
      value: 0.09,
  },
  {
      date: '2015',
      value: 0.13,
  },
  {
      date: '2016',
      value: 0.39,
  },
  {
      date: '2017',
      value: 1.00,
  },
  {
      date: '2018',
      value: 1.83,
  },
  {
      date: '2019',
      value:2.40,
  }]
    // Define SVG area dimensions
    const
        svgWidth = 600,
        svgHeight = 400;
  
    // Define the chart's margins as an object
    const margin = {
        top: 100,
        right: 60,
        bottom: 60,
        left: 60
    };
  
    // Define dimensions of the chart area
    const chartWidth = svgWidth - margin.left - margin.right;
    const chartHeight = svgHeight - margin.top - margin.bottom;
  
    // Select body, append SVG area to it, and set its dimensions
    const svg = d3.select("#line")
    .append("svg")
    .attr("width", svgWidth)
    .attr("height", svgHeight);
  
    // Append a group area, then set its margins
    const chartGroup = svg.append("g")
    .attr("transform", `translate(${margin.left}, ${margin.top})`);
  
    // Configure a parseTime function which will return a new Date object from a string
    const parseTime = d3.timeParse("%Y");
  
    // Load data from forcepoints.csv
    // Print the forceData
    console.log(data);
  
    // Format the date and cast the force value to a number
    data.forEach(function(data) {
        data.date = parseTime(data.date);
        data.value = +data.value;
    });
  
    // Configure a time scale
    // d3.extent returns the an array containing the min and max values for the property specified
    const xTimeScale = d3.scaleTime()
        .domain(d3.extent(data, d => d.date))
        .range([0, chartWidth]);
  
    // Configure a linear scale with a range between the chartHeight and 0
    const yLinearScale = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.value)])
        .range([chartHeight, 0]);
  
    // Create two new functions passing the scales in as arguments
    // These will be used to create the chart's axes
    const bottomAxis = d3.axisBottom(xTimeScale);
    const leftAxis = d3.axisLeft(yLinearScale);
  
    // Configure a line function which will plot the x and y coordinates using our scales
    const drawLine = d3.line()
        .x(data => xTimeScale(data.date))
        .y(data => yLinearScale(data.value));
  
    // Append an SVG path and plot its points using the line function
    chartGroup.append("path")
        // The drawLine function returns the instructions for creating the line for forceData
        .attr("d", drawLine(data))
        .classed("line", true);
  
    // Append an SVG group element to the chartGroup, create the left axis inside of it
    chartGroup.append("g")
        .classed("axis", true)
        .call(leftAxis);
  
    // Append an SVG group element to the chartGroup, create the bottom axis inside of it
    // Translate the bottom axis to the bottom of the page
    chartGroup.append("g")
        .classed("axis", true)
        .attr("transform", `translate(0, ${chartHeight})`)
        .call(bottomAxis);
      })()
