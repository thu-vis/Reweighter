<template>
<g :id="id" :transform="`translate(${x}, ${0})`">
    <g class="image-cluster-container" :transform="`translate(${containerPos}, 0)`">
      <g class="brush"></g>
      <rect class="image-cluster-background" fill="#fafafa" :width="clusterWidth+this.summaryWidth + this.hMargin+radius" :height="clusterHeight+2 * radius" :x="(cluster_type === `col` ? -containerPos: -radius)" :y="-radius"></rect>
      <g class="image-scatter" :transform="`translate(0, 0)`" v-if="mode == 'scatter'">
        <line :x1="0" :x2="clusterWidth" :y1="clusterHeight / 2" :y2="clusterHeight / 2" stroke="lightgrey" stroke-width="2" stroke-dasharray="2,2" opacity="0.5"/>
        <path class="trend" d="" stroke="lightgrey" stroke-width="1" fill="none" />
        <g v-if="lam_type===-1 && nrow>=1">
          <g v-for="d in imageScatterPoints" :key='`previous-${d.idx}`' :transform="`translate(${d.x},0)`">
            <!-- <g v-if="Math.abs(d.old_y - (d.y+d.p_offset+d.g_offset)) > 10"> -->
            <!-- <g v-if="d.diff"> -->
              <g v-if="all_samples[d.idx].diff">
              <circle class="previous"  cx=0 :cy="d.old_y" r="1" fill="#444444" fill-opacity="0.6"/>
            <line class="previous-line" :key='`previous-line-${d.idx}`' x1="0" x2="0" :y1="d.old_y" :y2="d.y+d.p_offset+d.g_offset"
            stroke-width=1 :stroke="d.old_y > d.y+d.p_offset+d.g_offset ? `GREEN` : `RED` "/>
            </g>
          </g>
        </g>
      </g>
    </g>

    <g class="image-cluster-info image-cluster-info-detailed" :transform="`translate(${infoPos}, 0)`">
      <ImageInfoGroup v-for="(group, i) in sample_images_group" :key='`ImageGroup-${id}-${i}`' :parentId='id' :group=group :is_val="cluster_type===`row`" :imgSize='imgSize'></ImageInfoGroup>
    </g>


    <g  class="cluster-summary" :transform="`translate(${summaryPos}, 0)`">
      <g v-show="nrow>=1" class="info-glyph" :transform="`translate(0, ${clusterHeight/2})`"></g>
      <g v-show="nrow<1" class="small-info-glyph" :transform="`translate(0, ${clusterHeight/2})`"></g>
      </g>

    <g v-if="cluster_type === `col`" :transform="`translate(${clickPos}, ${0})`">
      <!-- <path class="click-btn" fill="#444444" stroke="#eceeed" v-on:click="expandCluster" :transform="`translate(${0}, ${0})`"/>-->
      <rect fill="#ececec" :transform="`translate(${0}, ${0})`" witdh="8" height="8"/>
      <path class="click-btn" fill="#444444" stroke="#eceeed" v-on:click="expandCluster" :transform="`translate(${0}, ${0})`"/> 
    </g>
    
</g>
</template>

<script>
  import Vue from 'vue';
  import { contourDensity } from 'd3-contour';
  import { geoPath } from 'd3-geo';
  import { pie, arc } from 'd3-shape';
  import { select, selectAll } from 'd3-selection';
  import { brush, brushX, brushY } from 'd3-brush';
  import { drag } from 'd3-drag';
  import { mapState, mapActions, mapMutations } from 'vuex';
  import { get_inner_width, Shape, Color, get_polarity_color } from '@/plugins/utils.js';
  import { scaleLinear } from 'd3-scale';
  import { extent } from 'd3-array';

  const clamp = (num, min=-Infinity, max=Infinity) => Math.min(Math.max(num, min), max);

  export default Vue.extend({
    name: 'ImageCluster',
    components: {
        ImageInfoGroup: () => import('./image-info-group.vue'),
    },
    props: [
        'id',
        'showSummary',
        'info',
        'imgSize',
        'x',
        'y',
        'tagWidth',
        'infoWidth',
        'hMargin',
        'clusterWidth',
        'clusterHeight',
        'summaryWidth',
        'mode',
        'align',
        'ncol',
        'nrow'
    ],
    mounted: function() {
      this.node = select(`#${this.id}`);
      this.main_g = this.node.select(`.image-scatter`);
      this.resize();
      window.addEventListener('resize', () => {
        this.redrawToggle = false;
        setTimeout(() => {
          this.redrawToggle = true;
          this.resize();
        }, 300);
      });
      this.buildImageScatterPoints();
      this.setGroupMargin();
      this.setupBrush(null);
      // this.subcluster();

      this.$nextTick(function () {
        this.node.selectAll('.point.constraint').raise();
        this.generateDonutChart();
      })
      this.node.select('.click-btn').attr('d', this.nrow > 0.5 ? Shape.COLLAPSE_ICON : Shape.EXPAND_ICON)
      this.node.attr("transform", `translate(${this.x}, ${this.y})`);
    },
    updated() {
    },
    data() {
      return {
        redrawToggle: true,
        margin: 2,
        thinWidth: 2,
        boxWidth: 8,
        containerPos: 0,
        summaryPos: 0,
        infoPos: 0,
        clickPos: 0,
        expandIconPos: 0,
        selectionType: null,
        selection: null,
        selected: null,
        distribution: null,
        imageScatterPoints: null,
        imageScatterLine: null,
        radius: 3,
        brush2D: null,
        RED_BG: Color.RED_BG,
        GREEN_BG: Color.GREEN_BG,
        node: null,
        animation: {
          updateDuration: 1000,
        },
        summaryBg: '#E9EDED',
      }
    },
    methods: {
      ...mapMutations(['add_highlight_idx', 'rm_highlight_idx', 'update_color', 'set_update_cluster_by_idx']),
      ...mapActions(['fetch_sub_cluster']),
      resize() {
        this.svgWidth = get_inner_width(document.getElementById('image-grid-column'));
      },
      draw_density() {
        const density_data = this.info.all_index.map((idx, i) => {
          return {
            x: this.xScale(this.info.full_x[i]),
            y: this.yScale(this.info.full_w[i]),
          }
        })
        const contours = contourDensity()
          .x((d) => d.x)
          .y((d) => d.y)
          .size([this.clusterWidth, this.clusterHeight])
          .thresholds([0.4, 0.45, 0.5])
          .bandwidth(20)(density_data);

      this.node.append("g")
          .classed('density', true)
          .attr("fill", "none")
          .attr("stroke", "#fff")
          .attr("stroke-opacity", 0.5)
        .selectAll("path")
        .data(contours)
        .join("path")
          .attr("fill", d => "#ddd")
          .attr("d", geoPath());



      
      },
      setGroupMargin() {
        if (this.align == "left") {
          this.infoPos = 0;
          this.expandIconPos = this.infoPos - this.tagWidth;
          this.containerPos =  this.infoPos + this.infoWidth + this.hMargin;
          this.summaryPos = this.containerPos + this.clusterWidth + 1 * this.hMargin;
          this.clickPos = this.summaryPos + this.summaryWidth;
        } else {
          this.summaryPos = 0;
          this.containerPos = this.summaryPos + this.summaryWidth + this.hMargin;
          this.infoPos = this.containerPos + this.clusterWidth + this.hMargin;
          this.expandIconPos = this.infoPos + this.infoWidth;
          this.clickPos = -8;
        }
      },
      expandCluster() {
        this.$parent.toggleGroupExpand(this.info);
        this.node.select('.click-btn').attr('d', this.nrow > 0.5 ? Shape.COLLAPSE_ICON : Shape.EXPAND_ICON)
      },
      setupBrush(previousHeight) {
        this.node.selectAll(`.brush`).remove();
        if (this.mode === "scatter") {
          this.brush2D = brush()
              .extent([[0 - 1 * this.radius, 0 - this.radius], [this.clusterWidth + 2 * this.radius, this.clusterHeight + this.radius]])
              .on('start', event => {this.node.select('rect.selection').attr('fill', '#bbb');})
              .on('end', event => {
                this.selection = event.selection ? event.selection : null;
                this.selectionType = this.selection === null ? null : 'brush';
                this.updateSelection(null, null);
              });
          let gBrush = this.node.select(`.image-cluster-container`).append("g").attr("class", "brush").lower();
          this.node.selectAll('.image-cluster-background').lower();
          gBrush.call(this.brush2D);
          if (this.selectionType === 'brush' && this.selection !== null) {
            this.selection[0][1] *= this.clusterHeight / previousHeight;
            this.selection[1][1] *= this.clusterHeight / previousHeight;
            gBrush.call(this.brush2D.move, this.selection);
          }
        } else {
          this.selection = null;
        }
      },
      selectGroup(group, i) {
        if (this.selectionType === 'box' && this.selection === i) {
          this.selectionType = null;
          this.selection = null;
        } else {
          this.selectionType = 'box';
          this.selection = i;
        }
        this.updateSelection(null, null);
      },
      hoverGroup(g, i, hovered) {
        if (hovered) {
          this.updateSelection('box', i);
        } else {
          this.updateSelection(null, null);
        }
      },
      hoverSinglePoint(g, hovered) {
        if (hovered) {
          this.updateSelection('point', parseInt(g.currentTarget.id.split('-')[1]));
        } else {
          this.updateSelection(null, null);
        }
      },
      color_points(indices) {
        console.log('color points', indices);
        selectAll(`.point`).attr('fill', d => d.color);
        if (indices) {
          indices.forEach(idx => {
            select(`#point-${idx}`).attr('fill', Color.HIGHLIGHT);
          });
        }
      },
      highlight_points(indices) {
        console.log('highlight_points', indices);
        // this.node.selectAll(`.image-scatter .point`).style('opacity', 0.6);
        // this.node.selectAll(`.image-scatter .point`).attr('fill', d => d.color);
        // this.node.selectAll(`.image-scatter .vline`).style('opacity', 0.6);
        // if (indices) {
        //   indices.forEach(idx => {
        //     select(`#point-${idx}`).attr('fill', Color.HIGHLIGHT);
        //     // select(`#point-${this.id}-${idx}`).style('opacity', 1.0).attr('fill', Color.highlight);
        //     // select(`#vline-${this.id}-${idx}`).style('opacity', 1.0);
        //   });
        // }
        let previous_idx = this.highlight.map.get('cluster-' + this.id);
        if (previous_idx !== undefined) this.rm_highlight_idx({highlight: this.highlight, source: 'cluster-' + this.id});
        this.add_highlight_idx({highlight: this.highlight, source: 'cluster-' + this.id, idx: indices});
        // let previous_idx = this.highlight.map.get('cluster-' + this.id);
        // if (previous_idx !== undefined) {
        //   this.rm_highlight_idx({highlight: this.highlight, source: 'cluster-' + this.id});
        //   for (let idx of previous_idx) {
        //     this.rm_highlight_idx({highlight: this.high_pos_validation, source: this.id + '-sample-' + idx});
        //     this.rm_highlight_idx({highlight: this.high_neg_validation, source: this.id + '-sample-' + idx});
        //   }
        // }
        // if (indices) {
        //   if (this.cluster_type === 'row')  {
        //     let high_influence_idx = [];
        //     let pos_pairs = [], neg_pairs = [];
        //     for (let idx of indices) {
        //       let neg_list = this.all_samples[idx].sorted_influence_list.slice(0, 5), pos_list = this.all_samples[idx].sorted_influence_list.slice(this.all_samples[idx].sorted_influence_list.length-5);
        //       for (let col_idx of neg_list) neg_pairs.push([col_idx, idx])
        //       for (let col_idx of pos_list) pos_pairs.push([col_idx, idx])
        //       high_influence_idx.push(neg_list);
        //       high_influence_idx.push(pos_list);
        //     }
        //     high_influence_idx = [].concat(...high_influence_idx).unique();
        //     this.add_highlight_idx({highlight: this.highlight, source: 'cluster-' + this.id, idx: high_influence_idx});
        //     for (let col_idx of high_influence_idx) {
        //       this.add_highlight_idx({highlight: this.high_pos_validation, source: this.id + '-sample-' + col_idx, idx: pos_pairs.filter(d => d[0]===col_idx).map(d => d[1])});
        //       this.add_highlight_idx({highlight: this.high_neg_validation, source: this.id + '-sample-' + col_idx, idx: neg_pairs.filter(d => d[0]===col_idx).map(d => d[1])});
        //     }
        //   } else {
        //     this.add_highlight_idx({highlight: this.highlight, source: 'cluster-' + this.id, idx: indices});
        //     for (let idx of indices) {
        //       const scores = this.lam.map((l, ii) => l * this.R[ii][this.all_samples[idx].col_j]);
        //       const sorted_validation = scores.map((val,i) => {return {val,i}}).sort((a,b)=>b.val-a.val);
        //       this.add_highlight_idx({highlight: this.high_pos_validation, source: this.id + '-sample-' + idx, idx: sorted_validation.slice(0, 5).map(d => this.row_index[d.i])});
        //       this.add_highlight_idx({highlight: this.high_neg_validation, source: this.id + '-sample-' + idx, idx: sorted_validation.slice(sorted_validation.length-5).map(d => this.row_index[d.i])});
        //     }
        //   }
        // }
        this.$parent.highlightLinkFromIndex();
        this.$parent.updateValidation();
      },
      async subcluster() {
          let local_indices = this.info.local_indices;
          let n = Math.min(Math.max(Math.floor(Math.sqrt(local_indices.length)), 1), 7);
          if (n == this.info.sub_cluster_count) return;
          await this.fetch_sub_cluster({'cluster': this, 'idx': this.info.indices, 'local_idx': this.info.local_indices, 'n': n, 'type': this.cluster_type});
          this.buildDistribution();
          this.$nextTick(function () {
            this.setupGroupDrag();
          });
      },
      buildDistribution() {
        return;
        // let tmp = Array(this.info.sub_cluster_count).fill(null).map(() => Object({
        //   w: [],
        //   indices: [],
        //   constrained_indices: [],
        //   count: 0
        // }));
        // let points = this.info.local_indices.map((_, i) => i);
        // // if (this.selected === null) points = this.info.local_indices.map((_, i) => i); else points = this.selected;
        // points.forEach(pos => {
        //   let lbl = this.info.sub_cluster_label[pos];
        //   let idx = this.info.indices[pos];
        //   tmp[lbl].count += 1;
        //   tmp[lbl].w.push(this.info.w[pos]);
        //   tmp[lbl].indices.push(idx)
        //   if (this.all_samples[idx].is_pos || this.all_samples[idx].is_neg)
        //     tmp[lbl].constrained_indices.push(idx);
        // });

        // this.distribution = tmp.map(d => {
        //     return {
        //       w: d.w,
        //       upper: this.yScale(d.w.max()) - this.radius,
        //       lower: this.yScale(d.w.min()) + this.radius,
        //       offset: 0,
        //       offset_val: 0,
        //       indices: d.indices,
        //       constrained_indices: d.constrained_indices,
        //       color: Color.lightgrey,
        //       selected: false
        //     }
        //   });
      },
      buildImageScatterPoints() {
        let points = this.info.indices.map((idx, i) => Object({
            idx: idx,
            x: this.xScale(this.info.x[i]),
            y: this.yScale(this.info.w[i]),
            new_y: this.yScale(this.info.w_[i][1]),
            old_y: this.yScale(this.info.w_[i][0]),
            is_constraint: !this.is_row_cluster && (this.all_samples[idx].is_pos || this.all_samples[idx].is_neg),
            inconsistent: this.all_samples[idx].inconsistent,
            shape: this.all_samples[idx].shape,
            // shape: this.is_row_cluster ? Shape.CIRCLE : this.all_samples[idx].shape,
            // color: this.is_row_cluster ? Color.WHITE_POINT : this.all_samples[idx].color,
            color: this.all_samples[idx].color,
            label_color: this.all_samples[idx].label_color,
            p_offset: 0,
            g_offset: 0,
            val: this.info.w[i],
            old_val: this.info.w_[i][0],
            new_val: this.info.w_[i][1],
            p_offset_val: 0,
            g_offset_val: 0,
            diff: this.all_samples[idx].diff,
        }));
        points.sort((p, q) => p.x - q.x);
        this.imageScatterPoints = points;

        let points_g = this.main_g.selectAll('.point').data(this.imageScatterPoints, d=>d.idx);

        points_g.enter()
            .append('path')
            .attr('id', d => `point-${d.idx}`)
            .attr('class', d => d.is_constraint ? `point constraint` : `point`)
            .attr('transform', d => `translate(${d.x}, ${d.y+d.p_offset+d.g_offset})`)
            .attr('d', d => d.shape)
            .attr('fill', d => d.color)
            .on('mouseover', d => this.hoverSinglePoint(d, true))
            .on('mouseout', d => this.hoverSinglePoint(d, false))

        points_g.transition().duration(this.animation.updateDuration)
        .attr('transform', d => `translate(${d.x}, ${d.y+d.p_offset+d.g_offset})`)
        .attr('d', d => d.shape)
        .attr('fill', d => d.color)
        points_g.exit().remove();
        this.main_g.select(`.trend`).transition().duration(this.animation.updateDuration).attr('d', 'M'+this.imageScatterPoints.map(d => `${d.x},${d.y+d.p_offset+d.g_offset}`).join('L'));
        this.setupPointDrag();
      },
      setupPointDrag() {
        let is_row_cluster = this.is_row_cluster;
        let all_samples = this.all_samples;
        if (is_row_cluster) for (let point of this.imageScatterPoints) this.node.select(`#validation-${point.idx}`).raise()
        else for (let point of this.imageScatterPoints) this.node.select(`#point-${point.idx}`).raise()
        let that = this;
        this.node.selectAll(".point")
        .data(this.imageScatterPoints,d => d.idx)
        .call(drag()
        .on("start", (event, d) => {
          d.drag_y = event.y;
          d.previous_offset = d.p_offset;
          d.p_offset_max = this.clusterHeight - d.y - d.g_offset;
          d.p_offset_min = -d.y - d.g_offset;
        })
        .on("drag", (event, d) => {
          d.p_offset = d.previous_offset + event.y - d.drag_y;
          d.p_offset = clamp(d.p_offset, d.p_offset_min, d.p_offset_max);

          let distrib_pos = that.info.indices.indexOf(d.idx);
          let lbl = that.info.sub_cluster_label[distrib_pos];
          let distrib = that.distribution[lbl];
          let point_pos = distrib.indices.indexOf(d.idx);

          d.p_offset_val = this.yScaleBack(d.p_offset) - this.yScaleBack(0);
          let new_w = d.val + d.p_offset_val + d.g_offset_val;

          distrib.w[point_pos] = new_w;
          console.log(new_w);
          distrib.upper = that.yScale(distrib.w.max()) - that.radius;
          distrib.lower = that.yScale(distrib.w.min()) + that.radius;
          console.log(select(`point-${d.idx}`), d.y+d.p_offset+d.g_offset)
          this.main_g.select(`#point-${d.idx}`).attr('transform', `translate(${d.x}, ${d.y+d.p_offset+d.g_offset})`);
          this.main_g.select(`.trend`).attr('d', 'M'+this.imageScatterPoints.map(d => `${d.x},${d.y+d.p_offset+d.g_offset}`).join('L'));
          that.$forceUpdate();
        })
        .on("end", (event, d) => {
          d.drag_y = undefined;
          d.previous_offset = undefined;
          d.p_offset_max = undefined;
          d.p_offset_min = undefined;
          d.p_offset_val = this.yScaleBack(d.p_offset) - this.yScaleBack(0);
          console.log('drag end', d.val, d.p_offset_val, d.g_offset_val)
          if (is_row_cluster) all_samples[d.idx].set_lam = d.val + d.p_offset_val + d.g_offset_val;
          else {
            all_samples[d.idx].set_score = d.val + d.p_offset_val + d.g_offset_val;
            d.is_pos = d.is_up = (d.p_offset_val + d.g_offset_val > 0);
            d.is_neg = d.is_down = !d.is_up;
          }
        }));
      },
      setupGroupDrag() {
        let is_row_cluster = this.is_row_cluster;
        let all_samples = this.all_samples;
        this.node.selectAll(".summary-box")
        .data(this.distribution)
        .call(drag()
        .on("start", (event, d) => {
          d.drag_y = event.y;
          d.previous_offset = d.offset;
          let y_list = d.indices.map(idx => this.imageScatterPoints.find(dd => dd.idx === idx).y);
          d.offset_max = this.clusterHeight - y_list.max() - d.offset;
          d.offset_min = -y_list.min() - d.offset;
        })
        .on("drag", (event, d) => {
          d.offset = d.previous_offset + event.y - d.drag_y;
          d.offset = clamp(d.offset, d.offset_min, d.offset_max);
          d.indices.forEach(idx => {
            let point = this.imageScatterPoints.find(d => d.idx === idx);
            point.g_offset = d.offset;
          })
          this.main_g.selectAll('.point').attr('transform', d => `translate(${d.x}, ${d.y+d.p_offset+d.g_offset})`);
          this.main_g.select(`.trend`).attr('d', 'M'+this.imageScatterPoints.map(d => `${d.x},${d.y+d.p_offset+d.g_offset}`).join('L'));
        })
        .on("end", (event, d) => {
          d.drag_y = undefined;
          d.previous_offset = undefined;
          d.offset_val = this.yScaleBack(d.offset) - this.yScaleBack(0);
          d.indices.forEach(idx => {
            let point = this.imageScatterPoints.find(d => d.idx === idx);
            point.g_offset_val = this.yScaleBack(point.g_offset) - this.yScaleBack(0);
            if (is_row_cluster) all_samples[point.idx].set_lam = point.val + point.p_offset_val + point.g_offset_val;
            else {
              all_samples[point.idx].set_score = point.val + point.p_offset_val + point.g_offset_val;
              all_samples[point.idx].is_pos = all_samples[point.idx].is_up = (point.p_offset_val + point.g_offset_val) > 0;
              all_samples[point.idx].is_neg = all_samples[point.idx].is_down = !all_samples[point.idx].is_up;
            }
          })
        }));
      },
      updateSelection(selectionType, selection) {
        console.log('update selection', new Date().getTime())
        if (selectionType === null) {
          selectionType = this.selectionType;
          selection = this.selection;
        }
        console.log(selection, selectionType);
        let selectedIndices;
        // this.node.selectAll(`.cluster-summary rect`).attr("fill", Color.lightgrey);
        this.node.selectAll(`.pie`).attr("d", this.draw_arc);
        this.node.selectAll(`.gt-pie`).attr("d", this.gt_arc);
        if (selectionType === 'brush') {
          let selectedPoints = this.imageScatterPoints;
          selectedPoints = selectedPoints.filter(d => d.x >= this.selection[0][0] && d.x <= this.selection[1][0] && (d.y+d.p_offset+d.g_offset) >= this.selection[0][1] && (d.y+d.p_offset+d.g_offset) <= this.selection[1][1]);
          selectedIndices = selectedPoints.map(d => d.idx);
          this.color_points([]);
        } else if (selectionType === 'box') {
          this.node.select(`#${this.id}-summary-${selection} rect`).attr("fill", Color.highlight);
          selectedIndices = this.distribution[selection].indices;
          this.color_points(selectedIndices);
        } else if (selectionType === 'pie') {
          this.node.select(`#${this.id}-pie-${selection}`).attr("d", this.large_arc);
          this.node.select(`#${this.id}-gt-pie-${selection * 2}`).attr("d", this.large_gt_arc);
          this.node.select(`#${this.id}-gt-pie-${selection * 2 + 1}`).attr("d", this.large_gt_arc);
          selectedIndices = this.pie_data.find(d => d.index===selection).indices;
          this.color_points(selectedIndices);
        } else if (selectionType === 'bar') {
          selectedIndices = this.bar_data.find(d => d.index===selection).indices;
          this.color_points(selectedIndices);
        } else if (selectionType === 'point') {
          selectedIndices = [selection];
          this.color_points(selectedIndices);
        } else if (selectionType !== null) {
          console.log('unknown selection type: ', selectionType, selection);
          this.color_points([]);
        } else {
          this.color_points([]);
        }
        this.highlight_points(selectedIndices);
      },
      update_y: function() {
        this.imageScatterPoints.forEach(d => {
          d.g_offset = this.yScale(d.g_offset_val) - this.yScale(0);
          d.p_offset = this.yScale(d.p_offset_val) - this.yScale(0);
          d.new_y = this.yScale(d.new_val);
          d.old_y = this.yScale(d.old_val);
          d.y = this.lam_type===0 ? d.old_y : d.new_y;
        })
        // this.distribution.forEach(d => {
        //   d.offset = this.yScale(d.offset_val) - this.yScale(0);
        //   d.upper = this.yScale(d.w.max()) - this.radius;
        //   d.lower = this.yScale(d.w.min()) + this.radius;
        // })
      },
      generateDonutChart: function() {
        const that = this;
        if (this.is_row_cluster) {
        //   const mu = this.row_index.map(d => this.all_samples[d].lam).mean();
        //   const variance = this.row_index.map(d => this.all_samples[d].lam * this.all_samples[d].lam).sum() / this.row_index.length - mu * mu;
        //   const sd = Math.max(Math.sqrt(variance));
        //   // const sd = Math.max(Math.sqrt(variance), .5/this.row_index.length);
        //   // const sd = Math.sqrt(variance);
        //   const upper_threshold = Math.min(2*mu, mu + 1 * sd);
        //   const lower_threshold = Math.max(0.0001, mu - 1 * sd);
        //   const upper_index = this.info.indices.filter((idx) => (this.all_samples[idx].lam > upper_threshold));
        //   const middle_index = this.info.indices.filter((idx) => (lower_threshold <= this.all_samples[idx].lam && this.all_samples[idx].lam <= upper_threshold));
        //   const lower_index = this.info.indices.filter((idx) => (this.all_samples[idx].lam < lower_threshold));
        //   // console.log(this.info, mu, sd, upper_threshold, lower_threshold, upper_index, middle_index, lower_index)
        //   const data = [
        //     {index: 0, indices: upper_index, number: upper_index.length, color: Color.darkgrey},
        //     {index: 1, indices: middle_index, number: middle_index.length, color: Color.darkgrey},
        //     {index: 2, indices: lower_index, number: lower_index.length, color: Color.darkgrey},]
        //   const number_sum = data.map(d => d.number).sum()
        //   const g = this.node.select(".info-glyph").html(null);
        //   const bar_length = 35;
        //   const bar_height = this.clusterHeight / 6;

        //   // g
        //   // .selectAll('.validation-rect-line')
        //   // .data(data)
        //   // .enter()
        //   // .append('line')
        //   // .classed('validation-rect-line', true)
        //   // .attr('width', d => bar_length)
        //   // .attr('x1', d => 0)
        //   // .attr('x2', d => bar_length)
        //   // .attr('y1', (d, i) => this.clusterHeight * (i - 1) / 2.5)
        //   // .attr('y2', (d, i) => this.clusterHeight * (i - 1) / 2.5)
        //   // .attr('stroke', d => '#333333');
        //   g
        //   .selectAll('.validation-rect-bar')
        //   .data(data)
        //   .enter()
        //   .append('rect')
        //   .classed('validation-rect-bar', true)
        //   .attr('width', d => d.number == 0 ? 0 : 2 + d.number / (number_sum) * bar_length)
        //   .attr('height', bar_height)
        //   .attr('x', d => 0)
        //   .attr('y', (d, i) => this.clusterHeight * (i - 1) / 4 - bar_height/2)
        //   .attr('fill', d => d.color)
        //   .on('mouseover', (event, d) => {this.updateSelection('bar', d.index);})
        //   .on('mouseout', (event, d) => {this.updateSelection(null, null);})
        //   .on('click', (event, d) =>  {
        //     if (this.selectionType == 'bar' && this.selection == d.index) {
        //       this.selectionType = null;
        //       this.selection = null;
        //     } else {
        //       this.selectionType = 'bar';
        //       this.selection = d.index;
        //     }
        //     this.updateSelection(null, null);
        //   })
        //   .on('contextmenu', (event, d) => this.onContextmenu(event, d.indices));


        // // g
        // //   .selectAll('.validation-text-bg')
        // //   .data(data)
        // //   .enter()
        // //   .append('rect')
        // //   .classed('validation-text-bg', true)
        // //   .attr('fill', '#fafafa')
        // //   .attr('x', d => d.number == 0 ? 0 : 2 + d.number / (number_sum) * bar_length + 4)
        // //   .attr('y', (d, i) => this.clusterHeight * (i - 1) / 2.5 - bar_height/2)
        // //   .attr('width', d => d.number > 9 ? 10 : 5)
        // //   .attr('height', bar_height)
        // //   .attr('font-size', 10)
        // g
        //   .selectAll('.validation-text')
        //   .data(data)
        //   .enter()
        //   .append('text')
        //   .classed('validation-text', true)
        //   .text(d => d.indices.length)
        //   .attr('x', d => d.number == 0 ? 0 : 2 + d.number / (number_sum) * bar_length + 4)
        //   .attr('y', (d, i) => this.clusterHeight * (i - 1) / 4 - bar_height/2)
        //   .attr('dy', '7')
        //   .attr('font-size', 10)
          return;
        }
        // const idx_no_constraint = this.info.indices.filter((idx) => !(this.all_samples[idx].is_pos || this.all_samples[idx].is_neg));
        // const idx_pos_unsatisfied = this.info.indices.filter((idx, i) => this.all_samples[idx].is_pos && this.info.w[i] < 0);
        // const idx_neg_unsatisfied = this.info.indices.filter((idx, i) => this.all_samples[idx].is_neg && this.info.w[i] > 0);
        // const idx_pos_satisfied = this.info.indices.filter((idx, i) => this.all_samples[idx].is_pos && this.info.w[i] >= 0);
        // const idx_neg_satisfied = this.info.indices.filter((idx, i) => this.all_samples[idx].is_neg && this.info.w[i] <= 0);
        const g = this.node.select(".info-glyph").html(null);
        
        // const data = [
        //   {"index": 0, "pad": 0.01, "indices": idx_pos_satisfied, "number": idx_pos_satisfied.length+pie_offset, "name": "pos_satisfied", color: this.GREEN},
        //   {"index": 1, "pad": 0.01, "indices": idx_neg_unsatisfied, "number": idx_neg_unsatisfied.length+pie_offset, "name": "neg_unsatisfied", color: this.LIGHTRED},
        //   {"index": 2, "pad": 0.01, "indices": idx_pos_unsatisfied, "number": idx_pos_unsatisfied.length+pie_offset, "name": "pos_unsatisfied", color: this.LIGHTGREEN},
        //   {"index": 3, "pad": 0.01, "indices": idx_neg_satisfied, "number": idx_neg_satisfied.length+pie_offset, "name": "neg_satisfied", color: this.RED},
        //   {"index": 4, "pad": 0.01, "indices": idx_no_constraint, "number": idx_no_constraint.length+pie_offset, "name": "no_constraint", color: "#aaaaaa"}].filter(d => d.number>0);
        const consistent_thres = 1e-4;
        
        const idx_pos_consistent = this.info.indices.filter((idx, i) => !this.all_samples[idx].is_neg && this.info.w[i] > consistent_thres);
        const idx_pos_inconsistent = this.info.indices.filter((idx, i) => this.all_samples[idx].is_neg && this.info.w[i] > consistent_thres);
        const idx_neg_inconsistent = this.info.indices.filter((idx, i) => this.all_samples[idx].is_pos && this.info.w[i] < -consistent_thres);
        const idx_neg_consistent = this.info.indices.filter((idx, i) => !this.all_samples[idx].is_pos && this.info.w[i] < -consistent_thres);

        // const full_idx_pos_consistent = this.info.full_index.filter((idx, i) => !this.all_samples[idx].is_neg && this.all_samples[idx].full_score > consistent_thres);
        // const full_idx_pos_inconsistent = this.info.full_index.filter((idx, i) => this.all_samples[idx].is_neg && this.all_samples[idx].full_score > consistent_thres);
        // const full_idx_neg_inconsistent = this.info.full_index.filter((idx, i) => this.all_samples[idx].is_pos && this.all_samples[idx].full_score < -consistent_thres);
        const full_idx_pos_inconsistent = this.info.indices.filter((idx, i) => this.all_samples[idx].is_neg && this.all_samples[idx].full_score > consistent_thres);
        const full_idx_neg_inconsistent = this.info.indices.filter((idx, i) => this.all_samples[idx].is_pos && this.all_samples[idx].full_score < -consistent_thres);
        const full_idx_pos_consistent = this.info.full_index.filter((idx, i) => this.all_samples[idx].full_score > 0).filter(idx => !full_idx_pos_inconsistent.includes(idx));
        const full_idx_neg_consistent = this.info.full_index.filter((idx, i) => this.all_samples[idx].full_score < 0).filter(idx => !full_idx_neg_inconsistent.includes(idx));
        // const full_idx_neg_consistent = this.info.full_index.filter((idx, i) => !this.all_samples[idx].is_pos && this.all_samples[idx].full_score < -consistent_thres);

        // const min_num = Math.min(...[idx_pos_consistent.length, idx_pos_inconsistent.length, idx_neg_inconsistent.length, idx_neg_consistent.length])
        // const percentage = 1/15;
        // const pie_offset = Math.max((this.info.indices.length*percentage-min_num)/(1-4*percentage), 1);
        // const data = [
        //   {"index": 0, "pad": 0.01, "indices": idx_pos_consistent, "number": idx_pos_consistent.length+pie_offset, "name": "pos_consistent", color: this.GREEN, shape: Shape.CIRCLE},
        //   {"index": 1, "pad": 0.01, "indices": idx_pos_inconsistent, "number": idx_pos_inconsistent.length+pie_offset, "name": "pos_inconsistent", color: this.GREEN, shape: Shape.CROSS},
        //   {"index": 2, "pad": 0.01, "indices": idx_neg_inconsistent, "number": idx_neg_inconsistent.length+pie_offset, "name": "neg_inconsistent", color: this.RED, shape: Shape.CROSS},
        //   {"index": 3, "pad": 0.01, "indices": idx_neg_consistent, "number": idx_neg_consistent.length+pie_offset, "name": "neg_consistent", color: this.RED, shape: Shape.CIRCLE},
        //   ].filter(d => d.number>pie_offset);
        // const pie_offset = 0;
        // const data = [
        //   {"index": 0, "pad": 0.01, "indices": idx_pos_consistent, "number": idx_pos_consistent.length+pie_offset, "name": "pos_consistent", color: get_polarity_color(idx_pos_consistent.map(idx => this.all_samples[idx].color_value).mean(), 1), shape: Shape.CIRCLE},
        //   {"index": 1, "pad": 0.01, "indices": idx_pos_inconsistent, "number": idx_pos_inconsistent.length+pie_offset, "name": "pos_inconsistent", color: get_polarity_color(idx_pos_inconsistent.map(idx => this.all_samples[idx].color_value).mean(), 1), shape: Shape.CROSS},
        //   {"index": 2, "pad": 0.01, "indices": idx_neg_inconsistent, "number": idx_neg_inconsistent.length+pie_offset, "name": "neg_inconsistent", color: get_polarity_color(idx_neg_inconsistent.map(idx => this.all_samples[idx].color_value).mean(), -1), shape: Shape.CROSS},
        //   {"index": 3, "pad": 0.01, "indices": idx_neg_consistent, "number": idx_neg_consistent.length+pie_offset, "name": "neg_consistent", color: get_polarity_color(idx_neg_consistent.map(idx => this.all_samples[idx].color_value).mean(), -1), shape: Shape.CIRCLE},
        //   ]
          //.filter(d => d.number>pie_offset);
        const data = [
          {"index": 0, "pad": 0.01, "indices": idx_pos_consistent, "number": Math.pow(full_idx_pos_consistent.length,0.25), "display_number": full_idx_pos_consistent.length, "name": "pos_consistent", color: get_polarity_color(idx_pos_consistent.map(idx => this.all_samples[idx].color_value).mean(), 1), shape: Shape.CIRCLE},
          {"index": 1, "pad": 0.01, "indices": idx_pos_inconsistent, "number": Math.pow(full_idx_pos_inconsistent.length,0.25), "display_number": full_idx_pos_inconsistent.length, "name": "pos_inconsistent", color: get_polarity_color(idx_pos_inconsistent.map(idx => this.all_samples[idx].color_value).mean(), 1), shape: Shape.CROSS},
          {"index": 2, "pad": 0.01, "indices": idx_neg_inconsistent, "number": Math.pow(full_idx_neg_inconsistent.length,0.25), "display_number": full_idx_neg_inconsistent.length, "name": "neg_inconsistent", color: get_polarity_color(idx_neg_inconsistent.map(idx => this.all_samples[idx].color_value).mean(), -1), shape: Shape.CROSS},
          {"index": 3, "pad": 0.01, "indices": idx_neg_consistent, "number": Math.pow(full_idx_neg_consistent.length,0.25), "display_number": full_idx_neg_consistent.length, "name": "neg_consistent", color: get_polarity_color(idx_neg_consistent.map(idx => this.all_samples[idx].color_value).mean(), -1), shape: Shape.CIRCLE},
        ]
        const number_sum = data.map(d => d.number).sum();
        this.bar_data = data;
        // const arc_data = pie()
          // .value(d => d.number)
          // .padAngle((d,i) => 0.1)
          // .sort((a, b) => a.index-b.index)(data);
        // const draw_arc = arc()
        //     .innerRadius(7)
        //     .outerRadius(20);
        // const large_arc = arc()
        //     .innerRadius(7)
        //     .outerRadius(25);
        // const radius = 10 * Math.pow(this.info.indices.length, 0.1)
        // const draw_arc = arc()
        //     // .innerRadius(0.8*radius)
        //     // .outerRadius(1*radius);
        //     // .innerRadius(Math.min(...[0.9*radius, 8.2]))
        //     .cornerRadius((1.5+0.275*radius)/1.2)
        //     .innerRadius(radius-3)
        //     .outerRadius(1.55*radius);
        // const large_arc = arc()
        //     // .innerRadius(Math.min(...[0.9*radius, 8.2]))
        //     .innerRadius(radius-3)
        //     .outerRadius(1.1*radius);
        // const label_arc = arc()
        //     .innerRadius(1.18*radius)
        //     .outerRadius(1.22*radius);
        // this.draw_arc = draw_arc;
        // this.large_arc = large_arc;
        // this.label_arc = label_arc;
        // this.pie_data = this.bar_data = data;

        // g
        //   .selectAll('.pie')
        //   .data(arc_data)
        //   .enter()
        //   .append('path')
        //   .classed('pie', true)
        //   .attr('id', d => `${this.id}-pie-${d.data.index}`)
        //   .attr('d', draw_arc)
        //   // .attr('stroke', d => d.data.color)
        //   // .attr('stroke-width', 1)
        //   // .attr('fill', d => 'white')
        //   .attr('fill', d => d.data.color)
        //   .style('opacity', 1)
        //   .on('mouseover', (event, d) => {
        //     this.updateSelection('pie', d.data.index);
        //   })
        //   .on('mouseout', (event, d) => {
        //     this.updateSelection(null, null);
        //   })
        //   .on('click', (event, d) =>  {
        //     if (this.selectionType == 'pie' && this.selection == d.data.index) {
        //       this.selectionType = null;
        //       this.selection = null;
        //     } else {
        //       this.selectionType = 'pie';
        //       this.selection = d.data.index;
        //     }
        //     this.updateSelection(null, null);
        //   })
        //   .on('contextmenu', (event, d) => this.onContextmenu(event, d.data.indices));

        // g.append('text').text(this.info.indices.length)
        //   .attr('text-anchor', 'middle').attr('dy', 3.5).attr('font-size', this.info.indices.length>100 ? 10 : 12)
        // g
        //   .selectAll('.pie-glyph')
        //   .data(arc_data)
        //   .enter()
        //   .append('path')
        //   .classed('pie-glyph', true)
        //   .attr('d', d => d.data.shape)
        //   .attr('fill', d => 'white')
        //   // .attr('fill', d => d.data.color)
        //   // .attr('stroke', 'white')
        //   // .attr('stroke-width', 1)
        //   .attr('transform', function(d) {
        //     let pos = label_arc.centroid(d);
        //     return `translate(${pos[0]},${pos[1]})`;})
        // // g
        // //   .selectAll('.pie-arc')
        // //   .data(arc_data)
        // //   .enter()
        // //   .append('path')
        // //   .classed('pie-arc', true)
        // //   .attr('d', label_arc)
        // //   .attr('fill', d => d.data.color)

        const bar_length = this.is_row_cluster ? this.summaryWidth - 10 : this.summaryWidth - 25;

        const bar_height = this.clusterHeight/this.nrow / 6;
        g
          .selectAll('.large-rect-bar')
          .data(data)
          .enter()
          .append('rect')
          .classed('large-rect-bar', true)
          .attr('width', d => d.number  == 0 ? 0 : d.number / (number_sum) * bar_length)
          .attr('height', bar_height)
          .attr('x', d => 10)
          .attr('y', (d, i) => this.clusterHeight/this.nrow * (i - 1.5) /4 - bar_height/2)
          .attr('fill', d => d.color)
          .on('mouseover', (event, d) => {this.updateSelection('bar', d.index);})
          .on('mouseout', (event, d) => {this.updateSelection(null, null);})
          .on('click', (event, d) =>  {
            if (this.selectionType == 'bar' && this.selection == d.index) {
              this.selectionType = null;
              this.selection = null;
            } else {
              this.selectionType = 'bar';
              this.selection = d.index;
            }
            this.updateSelection(null, null);
          })
          .on('contextmenu', (event, d) => this.onContextmenu(event, d.indices));
          // .attr('stroke', d => d.color)
          // .attr('stroke-width', 1)
          // .attr('fill', d => d.color);
          // .attr('fill', d => d.index == 1 || d.index == 2 ? 'white' : d.color);

        g
          .selectAll('.bar-glyph-text')
          .data(data)
          .enter()
          .append('text')
          .classed('bar-glyph-text', true)
          .text(d => d.display_number)
          .attr('x', d => 10 + 2+d.number / (number_sum) * bar_length)
          .attr('y', (d, i) => this.clusterHeight/this.nrow * (i - 1.5)/4 - bar_height/2)
          .attr('dy', '8')
          .attr('font-size', 10)

        g
          .selectAll('.bar-glyph')
          .data(data)
          .enter()
          .append('path')
          .classed('bar-glyph', true)
          .style('paint-order', 'stroke')
          .attr('d', d => d.shape)
          .attr('fill', d => d.color)
          // .attr('stroke', this.summaryBg)
          // .attr('stroke-width',2)
          // .attr('fill', d => d.index==1||d.index==2?'white':d.color)
          .attr('transform', (d,i) => `translate(${5},${this.clusterHeight/this.nrow * (i - 1.5)/4})`)
        

        const small_g = this.node.select(".small-info-glyph").html(null);
        const pad = 0;
        // small_g
        //   .selectAll('.rect-bar-glyph')
        //   .data(data)
        //   .enter()
        //   .append('path')
        //   .classed('rect-bar-glyph', true)
        //   .attr('d', d => d.shape)
        //   .attr('fill', d => d.color)
        //   .attr('transform', d => `translate(${(d.number/2+data.filter(dd => dd.index < d.index).map(dd => dd.number).sum()) / (number_sum) * (this.imgSize - (data.length)* pad) + (data.filter(dd => dd.index < d.index).length+1)*pad},${this.clusterHeight/100})`)
        small_g
          .selectAll('.rect-bar')
          .data(data)
          .enter()
          .append('rect')
          .classed('rect-bar', true)
          .attr('width', d => d.number / (number_sum) * bar_length)
          .attr('height', this.clusterHeight*1.5)
          .attr('x', d => data.filter(dd => dd.index < d.index).map(dd => dd.number).sum() / (number_sum) * bar_length + (data.filter(dd => dd.index < d.index).length+1)*pad)
          .attr('y', -this.clusterHeight*0.75)
          // .attr('stroke', d => d.color)
          // .attr('stroke-width', 1)
          .attr('fill', d => d.color)
          .on('mouseover', (event, d) => {this.updateSelection('bar', d.index);})
          .on('mouseout', (event, d) => {this.updateSelection(null, null);})
          .on('click', (event, d) =>  {
            if (this.selectionType == 'bar' && this.selection == d.index) {
              this.selectionType = null;
              this.selection = null;
            } else {
              this.selectionType = 'bar';
              this.selection = d.index;
            }
            this.updateSelection(null, null);
          })
          .on('contextmenu', (event, d) => this.onContextmenu(event, d.indices));
          // .style('opacity', 1)
          // .on('mouseover', function(event, d){
          //   return;
          //   // const enlarge_pad=1;
          //   // select(this)
          //   //   .attr('width', d => 2*enlarge_pad+d.number / (that.info.indices.length) * (that.imgSize - (data.length) * pad))
          //   //   .attr('height', 2*enlarge_pad+that.clusterHeight+that.radius/2)
          //   //   .attr('x', d => -enlarge_pad+data.filter(dd => dd.index < d.index).map(dd => dd.number).sum() / (that.info.indices.length) * (that.imgSize - (data.length)* pad) + (data.filter(dd => dd.index < d.index).length+1)*pad)
          //   //   .attr('y', -enlarge_pad-that.radius/4)
          //   // that.updateSelection('bar', d.index);
          // })
          // .on('mouseout', function(event, d){ 
          //   select(this)
          //     .attr('width', d => d.number / (number_sum) * bar_length)
          //     .attr('height', this.clusterHeight)
          //     .attr('x', d => data.filter(dd => dd.index < d.index).map(dd => dd.number).sum() / (number_sum) * bar_length + (data.filter(dd => dd.index < d.index).length+1)*pad)
          //     .attr('y', -this.clusterHeight/2)
          //   that.updateSelection(null, null);
          // })
          // .on('click', function(event, d){ 
          //   if (that.selectionType == 'bar' && that.selection == d.data.index) {
          //     that.selectionType = null;
          //     that.selection = null;
          //   } else {
          //     that.selectionType = 'bar';
          //     that.selection = d.index;
          //   }
          //   that.updateSelection(null, null);
          // })
          

        const data_num = [this.info.full_index.length];
        small_g
          .selectAll('.small-bar-glyph-text')
          .data([data_num])
          .enter()
          .append('text')
          .classed('small-bar-glyph-text', true)
          .text(d => d)
          .attr('x', bar_length + 2)
          .attr('y', -this.clusterHeight/2)
          .attr('dy', '6')
          .attr('font-size', 10);



        // if (this.showgt) {
        //   const gt_data = [
        //     {"index": 0, "indices": idx_pos_satisfied.filter(i=>this.all_samples[i].correct), "number": idx_pos_satisfied.filter(i=>this.all_samples[i].correct).length, "name": "pos_satisfied", color: this.GREEN},
        //     {"index": 1, "indices": idx_pos_satisfied.filter(i=>!this.all_samples[i].correct), "number": idx_pos_satisfied.filter(i=>!this.all_samples[i].correct).length, "name": "pos_satisfied", color: this.RED},
        //     {"index": 2, "indices": idx_pos_unsatisfied.filter(i=>this.all_samples[i].correct), "number": idx_pos_unsatisfied.filter(i=>this.all_samples[i].correct).length, "name": "pos_unsatisfied", color: this.GREEN},
        //     {"index": 3, "indices": idx_pos_unsatisfied.filter(i=>!this.all_samples[i].correct), "number": idx_pos_unsatisfied.filter(i=>!this.all_samples[i].correct).length, "name": "pos_unsatisfied", color: this.RED},
        //     {"index": 4, "indices": idx_neg_satisfied.filter(i=>this.all_samples[i].correct), "number": idx_neg_satisfied.filter(i=>this.all_samples[i].correct).length, "name": "neg_satisfied", color: this.GREEN},
        //     {"index": 5, "indices": idx_neg_satisfied.filter(i=>!this.all_samples[i].correct), "number": idx_neg_satisfied.filter(i=>!this.all_samples[i].correct).length, "name": "neg_satisfied", color: this.RED},
        //     {"index": 6, "indices": idx_neg_unsatisfied.filter(i=>this.all_samples[i].correct), "number": idx_neg_unsatisfied.filter(i=>this.all_samples[i].correct).length, "name": "neg_unsatisfied", color: this.GREEN},
        //     {"index": 7, "indices": idx_neg_unsatisfied.filter(i=>!this.all_samples[i].correct), "number": idx_neg_unsatisfied.filter(i=>!this.all_samples[i].correct).length, "name": "neg_unsatisfied", color: this.RED},
        //     {"index": 8, "indices": idx_no_constraint.filter(i=>this.all_samples[i].correct), "number": idx_no_constraint.filter(i=>this.all_samples[i].correct).length, "name": "no_constraint", color: this.GREEN},
        //     {"index": 9, "indices": idx_no_constraint.filter(i=>!this.all_samples[i].correct), "number": idx_no_constraint.filter(i=>!this.all_samples[i].correct).length, "name": "no_constraint", color: this.RED}].filter(d => d.number>0);
        //   const gt_arc_data = pie()
        //     .value(d => d.number)
        //     .padAngle((d,i) => 0.05)
        //     .sort((a, b) => a.index-b.index)(gt_data);

        //   const gt_arc = arc()
        //       .innerRadius(21)
        //       .outerRadius(23);
        //   const large_gt_arc = arc()
        //     .innerRadius(26)
        //     .outerRadius(29);
        //   this.gt_arc = gt_arc;
        //   this.large_gt_arc = large_gt_arc;
        //   g
        //     .selectAll('.gt-pie')
        //     .data(gt_arc_data)
        //     .enter()
        //     .append('path')
        //     .classed('gt-pie', true)
        //     .attr('id', d => `${this.id}-gt-pie-${d.data.index}`)
        //     .attr('d', gt_arc)
        //     .attr('fill', d => d.data.color)
        //     // .style('opacity', 1);

        //   small_g
        //     .selectAll('.gt-rect-bar')
        //     .data(gt_data)
        //     .enter()
        //     .append('rect')
        //     .classed('gt-rect-bar', true)
        //     .attr('width', d => d.number / (this.info.indices.length) * (this.imgSize - (data.length) * pad))
        //     .attr('height', this.radius/4)
        //     // .attr('x', d => gt_data.filter(dd => dd.index < d.index).map(dd => dd.number).sum() / (this.info.indices.length) * (this.infoWidth - (data.length)* pad) + (gt_data.filter(dd => dd.index < d.index).length+1)*pad)
        //     .attr('x', d => gt_data.filter(dd => dd.index < d.index).map(dd => dd.number).sum() / (this.info.indices.length) * (this.imgSize - (data.length)* pad) + (gt_data.filter((dd, j) => dd.index <= d.index && (j === 0 || gt_data[j].name !== gt_data[j - 1].name)).length)*pad)
        //     .attr('y', this.radius*1.25)
        //     .attr('fill', d => d.color)
        //     // .style('opacity', 1);
        // }
      },

      onContextmenu(event, indices) {
        console.log(event, indices);
        const is_val = this.cluster_type===`row`
        if (is_val) {
          this.$contextmenu({
            items: [
          {
            label: "Relabel",
            icon: "icon-relabel",
            children: this.label_names.map((name, l)=>({
                label: name,
                onClick: () => {
                  this.onGroupChangeLabel(indices, l)
                },
                //icon: l == this.all_samples[this.idx].label ? "icon-check" : "",
              })
            ),
          },
          {
            label: "Remove",
            icon: "icon-remove",
            onClick: () => {this.onGroupRemove(indices)},
          },
          {
            label: "Increase weight",
            icon: "icon-good",
            onClick: () => {this.onGroupIncreaseWeight(indices)},
          },
          {
            label: "Decrease weight",
            icon: "icon-bad",
            onClick: () => {this.onGroupDecreaseWeight(indices)},
          },
        ],
            event,
            customClass: "custom-class",
            zIndex: 3,
            minWidth: 130
          });
        } else {
          this.$contextmenu({
            items: [
              {
                label: "Mark clean",
                icon: "icon-good",
                onClick: () => {this.onMarkClean(indices)}
              },
              {
                label: "Mark noisy",
                icon: "icon-bad",
                onClick: () => {this.onMarkNoisy(indices)}
              },
              {
                label: "Add into validation",
                icon: "icon-add",
                onClick: () => {this.onAddReward(indices)}
              }
            ],
            event,
            customClass: "custom-class",
            zIndex: 3,
            minWidth: 130
          });
        }
        event.preventDefault();
        return false;
      },
      onGroupChangeLabel(indices, l){
        console.log(`change label of ${indices} to ${l}`)
      },
      onGroupRemove(indices){
        console.log(`remove`, indices)
      },
      onGroupIncreaseWeight(indices) {
        console.log(`increase weights of`, indices)
      },
      onGroupDecreaseWeight(indices) {
        console.log(`decrease weights of`, indices)
      },
      onMarkClean(indices) {
        for (let idx of indices) {
          const sample = this.all_samples[idx];
          sample.is_pos = true;
          sample.is_neg = false;
          sample.color = Color.GREEN_POINT;
        }
      },
      onMarkNoisy(indices) {
        for (let idx of indices) {
          const sample = this.all_samples[idx];
          sample.is_pos = false;
          sample.is_neg = true;
          sample.color = Color.RED_POINT;
        }
      },
      async onAddReward(indices) {
        await this.fetch_relationship({'row_index': [indices]});
      },
    },
    computed: {
      ...mapState(['all_samples', 'color_scale', 'meta_data', 'row_index', 'highlight', 'high_pos_validation', 'high_neg_validation', 'R','lam','update_cluster_by_idx','redraw_cnt', 'lam_type', 'showgt', 'label_names']),
      GREEN() {
        return Color.GREEN_POINT;
      },
      RED() {
        return Color.RED_POINT;
      },
      LIGHTGREEN() {
        return Color.LIGHTGREEN;
      },
      LIGHTRED() {
        return Color.LIGHTRED;
      },
      cluster_type() {
        return this.id.substring(8, 11);
      },
      is_row_cluster() {
        return this.cluster_type==='row';
      },
      consistency_str() {
          return `${this.info.indices.filter(i => this.all_samples[i].gt == this.all_samples[i].pred).length}/${this.info.indices.length}`;
      },
      colCount() {
          return Math.floor(this.clusterWidth / this.imgSize);
      },
      sampled_images() {
        return this.info.indices.slice(-this.nrow * this.ncol);
        // return this.info.indices.slice(0, this.nrow * this.ncol);
        // return this.info.indices.sample(Math.min(this.nrow * this.ncol), this.info.indices.length);
      },
      sample_images_group() {
        let labels = this.info.indices.map(d => this.all_samples[d].label);
        let counter = new Array(this.label_names.length).fill(0);
        labels.forEach(d => counter[d]++);
        counter = counter.map((cnt,i) => ({cnt, i}))
                        .filter(d => d.cnt > 3)
                        .filter(d => d.cnt > labels.length / 5)
                        .sort((a,b) => (b.cnt-a.cnt))
                        .map(d => d.i)
                        .slice(0, 2);
        if (counter.length == 1) {
          return [{
            'label_name': this.label_names[counter[0]],
            'indices': this.info.indices.filter(d => this.all_samples[d].label==counter[0]).slice(0, 6),
            'cnt': 3,
            'x': 0,
            'margin': 0.2,
            'nrow': this.nrow,
          }];
        } else {
          return [
            {
            'label_name': this.label_names[counter[0]],
            'indices': this.info.indices.filter(d => this.all_samples[d].label==counter[0]).slice(0, 4),
            'cnt': 2,
            'x': 0,
            'margin': 0.05,
            'nrow': this.nrow,
          },
          {
            'label_name': this.label_names[counter[1]],
            'indices': this.info.indices.filter(d => this.all_samples[d].label==counter[1]).slice(0, 2),
            'cnt': 1,
            'x': 2.4,
            'margin': 0.1,
            'nrow': this.nrow,
          }]
        }

        // if (counter.length == 1) {
        //   return [{
        //     'label_name': this.label_names[counter[0]],
        //     'indices': this.info.indices.filter(d => this.all_samples[d].label==counter[0]).slice(0, 10),
        //     'cnt': 5,
        //     'x': 0,
        //     'margin': 0.2,
        //     'nrow': this.nrow,
        //   }];
        // } else {
        //   return [
        //     {
        //     'label_name': this.label_names[counter[0]],
        //     'indices': this.info.indices.filter(d => this.all_samples[d].label==counter[0]).slice(0, 6),
        //     'cnt': 3,
        //     'x': 0,
        //     'margin': 0.1,
        //     'nrow': this.nrow,
        //   },
        //   {
        //     'label_name': this.label_names[counter[1]],
        //     'indices': this.info.indices.filter(d => this.all_samples[d].label==counter[1]).slice(0, 3),
        //     'cnt': 2,
        //     'x': 3.7,
        //     'margin': 0.1,
        //     'nrow': this.nrow,
        //   }]
        // }
      },
      xScale() {
        return scaleLinear()
          .domain(extent(this.info.full_x))
          .range([0, this.clusterWidth]);
      },
      yScale() {
        if (this.info.type == 'row') {
          return scaleLinear()
            // .domain([0, Math.max(...[...this.info.full_w, 2.0 / this.row_index.length])])
            // .domain([0, 2.0 / this.row_index.length])
            .domain([-0.1/this.row_index.length, 2.1 / this.row_index.length])
            .range([this.clusterHeight,0]);
        } else {
          return scaleLinear()
            .domain([-this.info.w_.map(d => Math.max(Math.abs(d[0]), Math.abs(d[1]))).max(), this.info.w_.map(d => Math.max(Math.abs(d[0]), Math.abs(d[1]))).max()])
            //.domain([-this.info.w.map(d => Math.abs(d)).max(), this.info.w.map(d => Math.abs(d)).max()])
            .range([this.clusterHeight,0]);
        }
      },
      yScaleBack() {
        if (this.info.type == 'row') {
          return scaleLinear()
            .range([0, 2.0 / this.row_index.length])
            .domain([this.clusterHeight,0]);
        } else {
          return scaleLinear()
            .range([-this.info.w_.map(d => Math.max(Math.abs(d[0]), Math.abs(d[1]))).max(), this.info.w_.map(d => Math.max(Math.abs(d[0]), Math.abs(d[1]))).max()])
            .domain([this.clusterHeight,0]);
        }
      },
      highlight_idx() {
        return Array.from(this.highlight.idx).sort((a,b) => this.all_samples[a].score - this.all_samples[b].score);
      },
    },
    watch: {
      clusterHeight: function(newHeight, oldHeight) {
        // Animation
        this.node.style("opacity", 0)
        this.setupBrush(oldHeight);
        this.$parent.updateLinks();
        this.update_y();
        this.$nextTick(() => {
          this.node
            .transition()
            .duration(this.animation.updateDuration)
            .attr("transform", `translate(${this.x}, ${this.y})`)
            .style("opacity", 1);
          this.main_g.selectAll('.point').attr('transform', d => `translate(${d.x}, ${d.y+d.p_offset+d.g_offset})`);
          this.main_g.select(`.trend`).attr('d', 'M'+this.imageScatterPoints.map(d => `${d.x},${d.y+d.p_offset+d.g_offset}`).join('L'));
        });
      },
      y: function() {
        // Animation
        this.node
          .transition()
          .duration(this.animation.updateDuration)
          .attr("transform", `translate(${this.x}, ${this.y})`);
      },
      mode: function(newMode) {
        this.setupBrush(null);
      },
      update_cluster_by_idx: function(newVal) {
        if (this.info.indices.includes(newVal))  {
          this.buildImageScatterPoints();
          this.buildDistribution();
          this.setupPointDrag();
          // this.node.selectAll('.point.constraint').raise();
          this.generateDonutChart();
          this.$forceUpdate();
          this.set_update_cluster_by_idx(-1);
        }
      },
      redraw_cnt: function() {
          this.buildImageScatterPoints();
          this.buildDistribution();
          this.setupPointDrag();
          // this.node.selectAll('.point.constraint').raise();
          this.node.selectAll('.point').filter(d => d.inconsistent).raise();
          this.generateDonutChart();
          this.$forceUpdate();
      }
    }
  });
</script>

<style scoped>
.point {
  cursor: grab;
  /* opacity: 0.6; */
}
.summary-box {
  cursor: grab;
}
.vline {

  /* opacity: 0.6; */
}
.image-cluster-background {
  pointer-events: none;
}
.previous {
  pointer-events: none;
}
.previous-line {
  pointer-events: none;
}

rect.selection {
  fill: "#bbb";
}

</style>
