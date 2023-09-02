<template>
    <v-col cols='4' class='fill-width' id='confusion-matrix-column'>
    <v-col cols='12' class='topname fill-width'>Confusion Matrix</v-col>
    <svg v-if='redrawToggle === true' :width='svgWidth' :height='svgHeight' :viewBox='`${-margin.left},${-margin.top},${viewWidth+margin.left+margin.right},${viewHeight+margin.top+margin.bottom}`' id='confusion-matrix'></svg>
    </v-col>
</template>

<script>
  import Vue from 'vue'
  import { select } from 'd3-selection';
  import { mapState, mapActions, mapMutations } from 'vuex'
  import { to_percent, get_count, get_inner_width } from '@/plugins/utils.js';
  import { scaleLinear } from 'd3-scale';

  export default Vue.extend({
    name: 'ConfusionMatrix',
    mounted() {
      this.resize();
      this.update_view();
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
        viewWidth: 100,
        viewHeight: 100,
        margin: {'left': 5, 'right': 5, 'top': 5, 'bottom': 5},
        key1: 'gt',
        key2: 'label',
        color_scale: scaleLinear().domain([0, 0.1]).range(["#eff2f7", "#08326d"]),
        dark_font_color: '#08326d',
        light_font_color: '#eff2f7',
        small_split: 8,
      }
    },
    methods: {
    ...mapMutations(['add_highlight_idx', 'rm_highlight_idx']),
      get_pred_list(key) {
          if (key === 'gt') return this.samples.map(d => d.gt);
          if (key === 'label') return this.samples.map(d => d.label);
          if (this.meta_data.epochs.includes(key)) return this.preds[this.meta_data.epochs.indexOf(key)];
      },
      generate_confusion_matrix(groundtruth, pred) {
        let confusion_matrix = Array(this.meta_data.num_classes).fill(0).map(() => Array(this.meta_data.num_classes).fill(0));
        let idxes_matrix = Array(this.meta_data.num_classes).fill(0).map(()=>Array(this.meta_data.num_classes).fill(0).map(() => []))
        for (let i = 0; i < groundtruth.length; i++) {
          confusion_matrix[groundtruth[i]][pred[i]] += 1;
          idxes_matrix[groundtruth[i]][pred[i]].push(i);
        }
        return [confusion_matrix, idxes_matrix];
      },
      resize() {
        this.svgWidth = get_inner_width(document.getElementById('confusion-matrix-column'));
      },
      update_view() {
        console.log('update confusion matrix');
        const svg = select('#confusion-matrix');
        svg.selectAll("g.cell")
          .data(this.rect_data, d=>d.id)
          .join(
            enter => {
                const g = enter.append('g')
                .classed('cell', true)
                .attr('transform', d => `translate(${d.x},${d.y})`)
                .on('mouseover', this.mouseover_cell)
                .on('mouseout', this.mouseout_cell)
                g.append('rect')
                .classed('background', true)
                .attr('width', this.cell_size)
                .attr('height', this.cell_size)
                .attr('fill', d => this.color_scale(d.value))
                // g.append('rect')
                // .classed('overlay', true)
                // .attr('width', this.cell_size)
                // .attr('height', d => this.cell_size * d.ratio)
                // .attr('y', d => this.cell_size * (1-d.ratio))
                // .attr('fill', '#fc7551')
                g.append('text')
                .text(d => to_percent(d.value))
                .attr('fill', d => d.value < 0.03 ? this.dark_font_color : this.light_font_color)
                .attr('x', this.cell_size / 2)
                .attr('y', this.cell_size / 2)
                .attr('text-anchor', 'middle')
                .attr('font-size', this.cell_size / 5)
                .attr('dy', this.cell_size / 10)
            },
            update => {
                update.selectAll('rect.background')
                .attr('fill', d => this.color_scale(d.value));
                // update.selectAll('rect.overlay')
                // .attr('height', d => this.cell_size * d.ratio)
                // .attr('y', d => this.cell_size * (1-d.ratio))
                update.selectAll('text')
                .text(d => to_percent(d.value))
                .attr('fill', d => d.value < 0.025 ? this.dark_font_color : this.light_font_color)
            }
          );
      },
      update_overlay() {
          const svg = select('#confusion-matrix');
          svg.selectAll("g.cell").data(this.rect_data, d=>d.id).selectAll('rect.overlay').data(d => d.placeholder)
          .join(
            enter => {
                enter.append('rect')
                .classed('overlay', true)
                .attr('width', this.small_size)
                .attr('height', this.small_size)
                .attr('x', (d, i) => this.small_size * (i % this.small_split))
                .attr('y', (d, i) => this.small_size * (this.small_split - 1 - Math.floor(i / this.small_split)))
                .attr('fill', '#fc7551')
            },
            update => {
                update.selectAll('rect.overlay')
                .attr('x', (d, i) => this.small_size * (i % this.small_split))
                .attr('y', (d, i) => this.small_size * (this.small_split - 1 - Math.floor(i / this.small_split)))
            },
            exit => {
                exit.remove();
            }
          );
          svg.selectAll('text').raise();
      },
      mouseover_cell(event, d) {
          console.log(d, d.data)
          this.add_highlight_idx({highlight: this.highlight, source: 'confusion_matrix', idx: d.data});
      },
      mouseout_cell(event, d) {
          this.rm_highlight_idx({highlight: this.highlight, source: 'confusion_matrix'});
      },

    },
    computed: {
      ...mapState(['samples', 'preds', 'meta_data', 'highlight']),
      highlight_idx() {
        return this.highlight.idx;
      },
      arr1() {
        return this.get_pred_list(this.key1);
      },
      arr2() {
        return this.get_pred_list(this.key2);
      },
      rect_data() {
        const [confusion_matrix, idxes_matrix] = this.generate_confusion_matrix(this.arr1, this.arr2);
        console.log(confusion_matrix, idxes_matrix)
        const ret = [];
        for (let i = 0; i < this.meta_data.num_classes; ++i) {
            for (let j = 0; j < this.meta_data.num_classes; ++j) {
                ret.push({
                    id: `${i}-${j}`,
                    i: i,
                    j: j,
                    x: this.cell_size * i,
                    y: this.cell_size * j,
                    value: confusion_matrix[i][j] / this.samples.length,
                    gt: this.arr1[i],
                    label: this.arr2[j],
                    data: idxes_matrix[i][j],
                    ratio: 0,
                });
            }
        }
        return ret;
      },
    
      svgHeight() {
        return this.svgWidth;
      },
      cell_size() {
          return this.viewHeight / this.meta_data.num_classes;
      },
      small_size() {
        return this.cell_size / this.small_split;
      }
    },
    watch: {
        rect_data() {
            this.update_view();
        },
        highlight_idx() {
          this.rect_data.forEach(element => {
            element.ratio = get_count(element.data, this.highlight_idx) / 50;
            element.placeholder = Array(get_count(element.data, this.highlight_idx)).fill(0)
          });
          this.update_overlay();
        }
    }
  });
</script>

<style>
</style>