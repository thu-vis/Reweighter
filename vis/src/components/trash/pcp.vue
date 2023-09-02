<template>

<v-col cols='12' id='line-row'>
  <v-col cols='12' class='topname fill-width'>Weights</v-col>
    <svg v-if='redrawToggle === true' :width='svgWidth' :height='svgHeight' :viewBox='`0,0,${viewWidth},${viewHeight}`' id='pcp'>
      <g id="line-g"></g>
      <g id="axis-g"></g>
    </svg>
</v-col>
</template>

<script>
  import Vue from 'vue'
  import { select } from 'd3-selection';
  import { mapMutations, mapState } from 'vuex'
  import { scaleLinear, scalePoint, scaleOrdinal } from 'd3-scale';
  import { extent, ascending, cross } from 'd3-array';
  import { line } from 'd3-shape';
  import { axisRight } from 'd3-axis';
  import { brushY } from 'd3-brush';
  import Highlight  from '@/plugins/highlight.js';
  import { get_inner_width } from '@/plugins/utils.js';

  export default Vue.extend({
    name: 'PCP',
    components: {
    },
    mounted() {
      this.resize();
      window.addEventListener('resize', () => {
        this.redrawToggle = false;
        setTimeout(() => {
          this.redrawToggle = true;
          this.resize();
          this.update_view();
        }, 300);
      });
    },
    data() {
      return {
        svgWidth: 0,
        redrawToggle: true,
        viewWidth: 1400,
        viewHeight: 300,
        margin: {'left': 0, 'right': 0, 'top': 15, 'bottom': 5},
        brush_highlight: new Highlight(),
      }
    },
    methods: {
      ...mapMutations(['add_highlight_idx', 'rm_highlight_idx']),
      resize() {
        this.svgWidth = get_inner_width(document.getElementById('line-row'));
      },
      update_view() {
        if (!this.samples) return;
        const that = this;
        console.log('update pcp')
          select('#line-g')
            .attr('fill', 'none')
            .attr('stroke-width', 1)
            .selectAll('path.line')
            .data(this.samples, d => d.idx)
            .join('path')
            .classed('line', true)
            .attr('stroke', d => this.colors[d.idx])
            .attr('d', d => line()
            .defined(([, value]) => value != null)
            .x(([key, value]) => this.xScale(key))
            .y(([key, value]) => this.yScale.get(key)(value))(cross(this.pcp_attributes, [d], (key, d) => [key, d[key]])));

          
          const axes = select("#axis-g")
            .selectAll('g.axis')
            .data(this.pcp_attributes, d => d)
            .join(
            enter => enter.append('g')
            .classed('axis', true)
            .attr('transform', d => `translate(${this.xScale(d)},0)`)
            .each(function(d) { select(this).call( axisRight( that.yScale.get(d) ) ); } )
            .call(g => g.append('text')
                    .attr('x', -20)
                    .attr('y', 10)
                    .attr('text-anchor', 'start')
                    .attr('fill', 'black')
                    .text(d => d)),
            update => update.attr('transform', d => `translate(${this.xScale(d)},0)`),
            exit => exit.each(d => {console.log('exit', d)}).remove()
            )
          

          const updateBrushing = (attribute, selection) => {
            console.log('brushing');
            const values = this.yValue.get(attribute);
            this.lines.classed('hidden', (d) => {
              let path_visible = this.highlight.has(d.idx) && this.brush_highlight.has(d.idx);
              const [y0, y1] = selection;
              d.visible = path_visible && y0 <= values[d.idx] && values[d.idx] <= y1;
              return !d.visible;
            }).classed('visible', d=>d.visible)
            .filter(d => d.visible).raise();
          }

          const brushStart = (event, attribute) => {
            console.log('start')
            this.brush_highlight.rm_highlight(attribute);
            if (this.brush_highlight.is_empty()) this.rm_highlight_idx('brush');
            else this.add_highlight_idx({source: 'brush', idx: this.brush_highlight.idx});
            this.lines.classed('hidden', true).classed('visible', false).each(d => d.visible = false);
          }

          const brushed = (event, attribute) => {
            updateBrushing(attribute, event.selection);
          }

          const brushEnd = (event, attribute) => {
            console.log('brushend');
            if (event.selection === null) {
              return;
            }
            updateBrushing(attribute, event.selection);
            const values = this.yValue.get(attribute);
            const [y0, y1] = event.selection;
            this.brush_highlight.add_highlight(attribute, new Set(this.samples.filter((d,i) => y0 <= values[i] && values[i] <= y1).map(d => d.idx)));
            this.add_highlight_idx({source: 'brush', idx: this.brush_highlight.idx});
            this.update_color();
          }

          const brushes = axes.join('g.brush').classed('brush', true).call(
            brushY()
              .extent([[-10, this.margin.top / 2], [10, this.viewHeight]])
              .on('start', brushStart)
              .on('brush', brushed)
              .on('end', brushEnd)
          );
        this.update_color();
      },
      update_color() {
        console.log('update pcp color')
        this.lines
          .classed('hidden', d => !(d.visible = this.highlight.has(d.idx))).filter(d => d.visible).raise();
      }
    },
    computed: {
      ...mapState(['url', 'samples', 'weights', 'highlight', 'meta_data', 'colors', 'pcp_attributes', 'details']),
      highlight_idx() {
        console.log('pcp.vue highlight_idx');
        return this.highlight.idx;
      },
      lines () {
        console.log('pcp.vue lines');
        return select('#pcp').selectAll('path.line');
      },
      
      xScale() {
        console.log('pcp.vue xScale');
        return scalePoint(this.pcp_attributes, [this.margin.left, this.viewWidth - this.margin.right]);
      },

      
      yScale() {
        console.log('pcp.vue yScale');
        let scales = new Map();
        this.pcp_attributes.forEach(attribute => {
          if (attribute in ['gt', 'label', 'correct']) {
            scales.set(attribute,
              scalePoint(Array.from(new Set(this.samples.map(d => d[attribute]))).sort((a,b)=>a-b), [this.viewHeight - this.margin.bottom, this.margin.top])
            )
          } else {
            scales.set(attribute,
              scaleLinear()
                .range([this.viewHeight - this.margin.bottom, this.margin.top])
                .domain(extent(this.samples, item => item[attribute]))
              )
          }
        });
        console.log(scales)
        return scales;
      },
      yValue() {
        console.log('pcp.vue yValue');
        let values = new Map();
        this.pcp_attributes.forEach(attribute => {
          const y = this.yScale.get(attribute);
          values.set(attribute, this.samples.map(d => y(d[attribute])));
        });
        return values;
      },

      svgHeight() {
        return this.svgWidth / 5;
      }
    },
    watch: {
      pcp_attributes() {
        this.update_view();
      },
      highlight_idx() {
        this.update_color();
      }
    }
  });
</script>

<style>
path.line.hidden { stroke: #EEEEEE; }
</style>