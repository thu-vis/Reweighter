<template>
  <g :id="`histogram-range-${id}`" :transform="`translate(${x},${y})`">
    <rect v-for="d in bars" :id='`distribution-bar-${d.label}`' :key='`distribution-bar-${d.label}`' :x="d.x" :y="d.y" :width="d.width" :height="d.height" :fill="d.color"></rect>
  </g>
</template>

<script>
  import Vue from 'vue'
  import { mapState } from 'vuex'
  import { Shape, Color } from '@/plugins/utils.js';
  import * as d3Scale from 'd3-scale'
  import * as d3Array from 'd3-array'
  import * as d3Select from 'd3-selection'
  import * as d3Trans from 'd3-transition'
  import * as d3Brush from 'd3-brush'
  export default Vue.extend({
    name: 'ImageInfo',
    props: ['id','x','y','width','height','scores','index'],
    mounted() {
      const width = this.width - 20
      const min = d3Array.min(this.scores)
      const max = d3Array.max(this.scores)
      var histogram, bins, colors, brush
      this.updateBarColor = val => {
        var transition = d3Trans.transition().duration(this.transitionDuration)
        d3Trans
            .transition(transition)
            .selectAll(`.vue-histogram-slider-bar-${this.id}`)
            .attr('fill', d => {
            if (isTypeSingle) {
                return d.x0 < val.from ? colors(d.x0) : this.holderColor
            }
            return d.x0 <= val.to && d.x0 >= val.from ? colors(d.x0) : this.holderColor
            })
        }
        let x = d3Scale
        .scaleLinear()
        .domain([min, max])
        .range([0, width])
        .clamp(true)
        let y = d3Scale.scaleLinear().range([this.height, 0])
        let g = d3Select.select(`#histogram-range-${id}`)
        const updateHistogram = ([min, max]) => {
        let transition = d3Trans.transition().duration(this.transitionDuration)
        hist.selectAll(`.vue-histogram-slider-bar-${this.id}`).remove()
        histogram = d3Array
            .bin()
            .domain(x.domain())
            .thresholds(width / (this.barWidth + this.barGap))
        // group data for bars
        bins = histogram(this.data)
        y.domain([0, d3Array.max(bins, d => d.length)])
        hist
            .selectAll(`.vue-histogram-slider-bar-${this.id}`)
            .data(bins)
            .enter()
            .insert('rect', 'rect.overlay')
            .attr('class', `vue-histogram-slider-bar-${this.id}`)
            .attr('x', d => x(d.x0))
            .attr('y', d => y(d.length))
            .attr('width', this.barWidth)
            .transition(transition)
            .attr('height', d => this.barHeight - y(d.length))
            .attr('fill', d => (isTypeSingle ? this.holderColor : colors(d.x0)))
        }
        updateHistogram([min, max])
    },
    computed: {
      ...mapState(['all_samples', 'meta_data', 'color_map_list', 'redraw_cnt']),
    },
    methods: {
    },
    watch: {
    }

});
</script>


<style>

</style>
