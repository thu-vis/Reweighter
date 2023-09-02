<template>
    <v-col cols='12' class='fill-width' id='bipartite'>
    <!-- <v-col cols='12' class='topname fill-width'>Cluster level</v-col> -->
    <div class='topname fill-width'>Cluster level</div>
    <div cols='12' class='h-line fill-width'></div>
    <div>
      
      <button style="float: right; margin-left: 32px; color: dark;" @click="fetch_opt_result">Update</button>
      <button id="mode-selection-diff" class="mode-selection" style="float: right; margin-left: 8px; color: lightgrey;" @click="showDiff" >Diff</button>
      <button id="mode-selection-current" class="mode-selection" style="float: right; margin-left: 8px; color: dark;" @click="showCurrent" >Current</button>
      <!-- <button id="mode-selection-previous" class="mode-selection" style="float: right; margin-left: 8px; color: lightgrey;" @click="showPrevious">Previous</button> -->
      <span style="float: right; margin-right: 8px"  > 
        Diff threshold
      <input id="threshold-slider" type="range" min="0" max="1" value="0.1" step="0.01" @change="updateThreshold" v-model="threshold"><!---->
      </span>
      <span style="float: right; margin-right: 8px"  > 
        gamma
      <input id="gamma-slider" type="range" min="0" max="1" value="0.1" step="0.01" @change="updateGamma" v-model="gamma"><!---->
      </span>
      <button id="save" style="float: right; margin-left: 8px; color: white;" @click="save">Save</button>
    </div>

    <svg :width='imageClusterWidth' :height='viewHeight' @wheel="onWheel" viewBox="0 0 1600 1200">
      <g id="staticlegend" x="0" y="0" >
        <!-- <rect fill="#ffffff" x="0" y="0" :width='imageClusterWidth' :height='230'></rect> -->
        <rect fill="#f5f6f9" x="50" y="0" :width='1500'  height="75" rx="10" ry="10"></rect>
        <AILegend/>
      </g>
      <transition appear name="image-cluster-group-transition"
        enter-active-class="animate__animated animate__fadeIn"
        leave-active-class="animate__animated animate__fadeOut">
      <g id="bipartite-group" :transform='`translate(75, 0)`'>
        <defs v-if="row_group">
          <clipPath id="link-filter">
            <rect :x="(clusterRightStart - linkMargin) - thumbnail_width"
                  :y="0"
                  :width="thumbnail_width"
                  :height="viewHeight"/>
            <rect :x="clusterLeftEnd + linkMargin"
                  :y="0"
                  :width="thumbnail_width"
                  :height="viewHeight"/>
          </clipPath>
          <clipPath id="link-dash-clippath">
            <rect :x="(clusterRightStart - linkMargin) - thumbnail_width"
                  :y="0"
                  :width="thumbnail_width"
                  :height="viewHeight"/>
            <rect :x="clusterLeftEnd + linkMargin"
                  :y="0"
                  :width="thumbnail_width"
                  :height="viewHeight"/>
            <rect v-for="idx in [0,1,2,3,4]" :key="`clipPathRect-${idx}`" :x="clusterLeftEnd + linkMargin + thumbnail_width + (2*idx+1) * ((clusterRightStart - linkMargin) - (clusterLeftEnd + linkMargin) - 2 * thumbnail_width) / 11"
                  :y="0"
                  :width="((clusterRightStart - linkMargin) - (clusterLeftEnd + linkMargin) - 2 * thumbnail_width) / 11"
                  :height="viewHeight"/>
          </clipPath>
          <clipPath id="bipartiteClipCircle">
            <circle :r="imgSize/2" :cx="imgSize/2" :cy="imgSize/2"/>
          </clipPath>


          <filter x="0" y="0" width="1" height="1" id="text-bg">
            <feFlood flood-color="#fafafa" result="bg" />
            <feMerge>
              <feMergeNode in="bg"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>
        
        <text id="row-group-title" :x='clusterLeftStart +(tagWidth+leftClusterWidth+leftInfoWidth)/2'  :y='0' font-size=1rem text-anchor='middle'>Validation samples</text>
        <text id="col-group-title" :x='clusterRightStart+(tagWidth+rightClusterWidth+rightInfoWidth)/2' :y='0' font-size=1rem text-anchor='middle'>Training samples</text>
        <g id='row-groups'>
        <ImageCluster v-for="(cluster) in row_group" :id='`cluster-row-${cluster.id}`' :key='`cluster-row-${cluster.id}`' :info='cluster' :x='clusterLeftStart' :y='cluster.offset' :imgSize='imgSize' :clusterWidth='leftClusterWidth' :clusterHeight='cluster.nrow * clusterHeight'
         :tagWidth='tagWidth' :infoWidth='leftInfoWidth' :hMargin='hMargin' :summaryWidth='leftSummaryWidth' :showSummary='true' :mode="mode" :nrow="cluster.nrow" :ncol="5" align="left"/>
        </g>
        <g id='col-groups'>
        <ImageCluster v-for="(cluster) in col_group" :id='`cluster-col-${cluster.id}`' :key='`cluster-col-${cluster.id}`' :info='cluster' :x='clusterRightStart' :y='cluster.offset' :imgSize='imgSize' :clusterWidth='rightClusterWidth' :clusterHeight='cluster.nrow * clusterHeight'
         :tagWidth='tagWidth' :infoWidth='rightInfoWidth' :hMargin='hMargin' :summaryWidth='rightSummaryWidth' :showSummary='true' :mode="mode" :nrow="cluster.nrow" :ncol="5" align="right"/>
        </g>
        <g class='link-group'>
        <!-- <path v-for="(link, i) in links" :key='`link(${i})`' :class="link.running ? `running link`: `link`" :d='link.d' :fill="weightColor(link.influence)" :stroke="weightStroke(link.influence)" :opacity='0.5'
         :clip-path="link.display ? link.influence > 0 ? `` : `url(#link-dash-clippath)` : `url(#link-filter)`" @mousemove="linkMouseMove(link, $event)" @mouseout="linkMouseOut(link, $event)" @click="linkMouseClick(link, $event)"/>-->
        <!-- <path v-for="(link) in links" :key='`link(${link.row}-${link.col})`' :class="link.running ? `running link`: `link`" d='' :fill="weightColor(link.influence)" :stroke="weightStroke(link.influence)" :opacity='0.5'
         :clip-path="link.display && ((highlight_row_index.length+highlight_col_index.length)==0 || col_group[link.col].nrow>=1) ? `` : `url(#link-filter)`" @mouseover="linkMouseMove(link, $event)" @mouseout="linkMouseOut(link, $event)" @click="linkMouseClick(link, $event)"/>
         -->
        <!-- <path v-for="(link) in links" :key='`bg-link(${link.row}-${link.col})`' :class="link.running ? `running bglink`: `bglink`" d='' :fill="weightColor(link.influence)" :stroke="weightStroke(link.influence)" :opacity='0.2' @mouseover="linkMouseMove(link, $event)" @mouseout="linkMouseOut(link, $event)" @click="linkMouseClick(link, $event)"/> -->
        <path v-for="(link) in links" :key='`bg-link(${link.row}-${link.col})`' :id="`bg-link(${link.row}-${link.col})`" :class="link.display ? `bglink colored-link` : link.exist ? `bglink` : `bglink invisible-link`" :d='link.d' :fill="link.display ? weightStroke(link.influence) : `#aaaaaa`"  :opacity="0.25" @mouseover="linkMouseMove(link, $event)" @mouseout="linkMouseOut(link, $event)" @click="linkMouseClick(link, $event)"/>
        </g>
        <g class='link-core-group'>   
          <!-- <path v-for="(link) in links" :key='`bg-link(${link.row}-${link.col})`' :class="link.running ? `running bglink`: `bglink`" d='' :fill="weightColor(link.influence)" :stroke="weightStroke(link.influence)" :opacity="link.display && ((highlight_row_index.length+highlight_col_index.length)==0 || col_group[link.col].nrow>=1)? 0.5 : 0"/> -->
          <!-- <path v-for="(link) in links" :key='`link-core(${link.row}-${link.col})`' :id="`link-core(${link.row}-${link.col})`" :class="link.display ? `link-core colored-link-core` : `link-core`" d='' fill="none" :stroke="weightStroke(link.influence)" :opacity="link.display ? 1 : 0" pointer-events="none" stroke-linecap="butt" stroke-linejoin="round"  stroke-width="2"/>-->
<!--:stroke-dasharray="link.influence > 0 ? `` : `2,2`"
link.display && ((highlight_row_index.length+highlight_col_index.length)==0 || col_group[link.col].nrow>=1)? 0.5 : 0.25-->
        </g>
      </g>
      </transition>


      <!-- <g id="legend">
        <rect fill="#ffffff" x="0" y="0" :width='600' :height='230'></rect>
        <rect fill="#f5f6f9" x="10" y="0" :width='600'  height="210" rx="10" ry="10"></rect>
      </g> -->

      <!-- <Distribution :x='clusterRightEnd - distribution_width' :y='0' :width='distribution_width' :height='210'></Distribution> -->

    </svg>
    </v-col>
</template>

<script>
  import Vue from 'vue'
  import { pie, arc } from 'd3-shape';
  import { select, selectAll } from 'd3-selection';
  import { mapState, mapActions, mapMutations } from 'vuex';
  import { get_inner_width, Color } from '@/plugins/utils.js';
  import { scaleLinear } from 'd3-scale';
  import * as d3 from 'd3';
  import * as reorder from 'reorder.js';

  export default Vue.extend({
    name: 'Bipartite',
    components: {
        ImageCluster: () => import('./image-cluster.vue'),
        AILegend: () => import('./ai-legend.vue'),
        // Distribution: () => import('./distribution.vue'),
    },
    mounted() {
      window.bipartite = this;
      this.resize();
      window.addEventListener('resize', () => {
        this.redrawToggle = false;
        setTimeout(() => {
          this.redrawToggle = true;
          this.resize();
        }, 300);
      });
      // document.getElementById("threshold-slider").value = 0.1;
      // document.getElementById("gamma-slider").value = 0.1;
      // this.drawLegend2();
      console.log("mount bipartite");
    },
    data() {
      let tmp = {
        tagWidth: 0,
        imageClusterWidth: 0,
        redrawToggle: true,

// display
        leftClusterWidth: 250,
        rightClusterWidth: 500,
        clusterMargin: 16,
        leftSummaryWidth: 10,
        rightSummaryWidth: 100,
        linkDirect: 20,
        linkMargin: 25,
        linkWidth: 250,

        // leftClusterWidth: 250,
        // rightClusterWidth: 500,
        // clusterMargin: 16,
        // leftSummaryWidth: 70,
        // rightSummaryWidth: 100,
        // linkDirect: 20,
        // linkMargin: 25,
        // linkWidth: 250,

        // teaser
        // leftClusterWidth: 180,
        // rightClusterWidth: 360,
        // clusterMargin: 16,
        // leftSummaryWidth: 0,
        // rightSummaryWidth: 60,
        // linkDirect: 10,
        // linkMargin: 15,
        // linkWidth: 150,

        
        // leftClusterWidth: 160,
        // rightClusterWidth: 240,
        // clusterMargin: 16,
        // leftSummaryWidth: 0,
        // rightSummaryWidth: 40,
        // linkDirect: 10,
        // linkMargin: 15,
        // linkWidth: 80,

        clusterHeight: 36,
        imgSize: 32,

        // leftClusterWidth: 150,
        // rightClusterWidth: 180,
        // clusterMargin: 15,
        // leftSummaryWidth: 50,
        // rightSummaryWidth: 50,
        // linkDirect: 0,
        // linkMargin: 15,
        // linkWidth: 120,

        leftInfoWidth: 130,
        rightInfoWidth: 150,
        hMargin: 10,
        margin: {'left': 0, 'right': 0, 'top': 15, 'bottom': 5},
        links: [],

        thumbnail_width: 30,
        
        singleAdjustments: Object(),
        clusterAdjustments: Object(),
        mode: "scatter",
        value_threshold: 0.5,
        ratio_threshold: 0.3,

        linkStatus: null,
        clickStatus: null,

        col_group: null,
        row_group: null,

        viewHeight: 1,
        reorder: null,

        distribution_width: 500,
        
        scrollTranslate: 0,
        // bipartiteTransalate: 30 + 210 + 30,
        bipartiteTransalate: 30 + 0 + 30,
        rowGroupTranslate: 0,
        colGroupTranslate: 0,

        influence_ratio: 20,
        gamma: 0.1,
        threshold: 0.1,
      }
      tmp.clusterLeftStart = tmp.tagWidth;
      tmp.clusterLeftEnd = tmp.clusterLeftStart + tmp.leftInfoWidth + tmp.hMargin + tmp.leftClusterWidth + tmp.hMargin + tmp.leftSummaryWidth;
      tmp.clusterRightStart = tmp.clusterLeftEnd + tmp.linkWidth + 2 * tmp.linkMargin;
      tmp.clusterRightEnd = tmp.clusterRightStart + tmp.rightInfoWidth + tmp.hMargin + tmp.rightClusterWidth + tmp.hMargin + tmp.rightSummaryWidth;
      return tmp;
    },
    methods: {
      ...mapActions(['fetch_opt_result', 'run', 'save']),
      ...mapMutations(['set_matrix', 'redraw_all', 'choose_lam', 'set_lam_type', 'set_gamma', 'set_threshold']),
      resize() {
        let svgWidth = get_inner_width(document.getElementById('bipartite'));
        this.imageClusterWidth = svgWidth;
      },
      showPrevious() {
        selectAll(".mode-selection").style("color", "lightgrey");
        select(`#mode-selection-previous`).style("color", "black");
        this.set_lam_type(0);
        this.redraw_all();
      },
      showCurrent() {
        selectAll(".mode-selection").style("color", "lightgrey");
        select(`#mode-selection-current`).style("color", "black");
        this.set_lam_type(1);
        this.redraw_all();
      },
      showDiff() {
        selectAll(".mode-selection").style("color", "lightgrey");
        select(`#mode-selection-diff`).style("color", "black");
        this.set_lam_type(-1);
        this.redraw_all();
      },
      toggleMode(mode) {
        this.mode = mode;
        selectAll(".mode-selection").style("color", "lightgrey");
        select(`#mode-selection-${mode}`).style("color", "black");
      },
      updateGroupHeight() {
        this.col_group.forEach(g => {
          g.offset = this.col_group.filter(d => d.order < g.order).map(d => this.clusterHeight * d.nrow + this.clusterMargin).sum()
        })
        this.row_group.forEach(g => {
          g.offset = this.row_group.filter(d => d.order < g.order).map(d => this.clusterHeight * d.nrow + this.clusterMargin).sum()
        })
        let col_total_offset = this.col_group.map(d => this.clusterHeight * d.nrow + this.clusterMargin).sum();
        let row_total_offset = this.row_group.map(d => this.clusterHeight * d.nrow + this.clusterMargin).sum();
        // let col_total_offset;
        // col_total_offset = 0;
        // this.col_group.forEach(g => {
        //   g.offset = col_total_offset;
        //   col_total_offset += (this.clusterHeight) * g.nrow + this.clusterMargin;
        // })
        // let row_total_offset = 0;
        // this.row_group.forEach(g => {
        //   g.offset = row_total_offset;
        //   row_total_offset += (this.clusterHeight) * g.nrow + this.clusterMargin;
        // })
        console.log('update height', row_total_offset,col_total_offset);
        if (row_total_offset > col_total_offset) {
          this.col_group.forEach(g => {g.offset += (row_total_offset-col_total_offset)/2;})
        } else {
          this.row_group.forEach(g => {g.offset += (col_total_offset-row_total_offset)/2;})
        }
        if (this.row_group[0].offset < 250) {
          let offset = 250 - this.row_group[0].offset;
          this.col_group.forEach(g => {g.offset += offset;})
          this.row_group.forEach(g => {g.offset += offset;})
        }
        // if (Math.max(row_total_offset, col_total_offset) < 1000) {
        //   let offset = (1000 - Math.max(row_total_offset, col_total_offset)) / 2;
        //   this.col_group.forEach(g => {g.offset += offset;})
        //   this.row_group.forEach(g => {g.offset += offset;})
        // }
        //this.viewHeight = Math.max(this.clusterHeight*this.row_group.map(d => d.nrow).sum()+(this.row_group.length)*this.clusterMargin,
        //               this.clusterHeight*this.col_group.map(d => d.nrow).sum()+(this.col_group.length)*this.clusterMargin);
        
        this.bipartiteTransalate = this.margin.top * 2 + Math.max(0, 0 + 30 - Math.min(this.row_group ? this.row_group.map(d => d.offset).min() : 0, 0 + 30 - this.col_group ? this.col_group.map(d => d.offset).min() : 0));
        this.viewHeight = 1000;//[...this.row_group.map(d => d.offset+d.nrow*this.clusterHeight), ...this.col_group.map(d => d.offset+d.nrow*this.clusterHeight)].max()+this.bipartiteTransalate + 10;
        select("#bipartite-group")
          .transition()
          .duration(1000)
          .attr("transform", `translate(75, ${this.bipartiteTransalate})`);
        // this.$forceUpdate();
        this.animateGroupTitle();
      },
      toggleGroupExpand(group) {
        // group.isFocus = !group.isFocus;
        // group.nrow = group.isFocus ? 2 : this.get_nrow(group);
        // this.updateGroupHeight();
        // group.isFocus = !group.isFocus;
        // group.nrow = group.isFocus ? 1 : 0.1;
        group.nrow = group.nrow == 1 ? 0.1 : 1;
        this.updateGroupHeight();
      },
      weightColor(weight) {
        // let polarity = weight > 0 ? "#47be3d" : "#d03a19",
        //     white = "#ffffff";
        // return d3.interpolate(white, polarity)(Math.max(0.2, Math.abs(weight)));
        // return "#f4f4f7";
        // return "#d0cedd";
        if (weight > 0) return Color.LIGHTGREEN;
        if (weight < 0) return Color.LIGHTRED;
      },
      weightStroke(weight) {
        if (weight > 0) return Color.GREEN_POINT;
        if (weight < 0) return Color.RED_POINT;
        // return "#d0cedd";
        // return weight > 0 ? Color.GREEN_POINT : Color.RED_POINT;
      },
      updateLinks() {
        if (!this.row_group || !this.col_group) return;
        const logic_row_group = Array.from(Array(this.num_row_cluster).keys()).map(i => this.row_group.find(d => d.order==i));
        const logic_col_group = Array.from(Array(this.num_col_cluster).keys()).map(i => this.col_group.find(d => d.order==i));
        let linkH = d3.linkHorizontal().x(d => d.x).y(d => d.y);
        const linkStart = this.clusterLeftEnd + this.linkMargin;
        const linkEnd = this.clusterRightStart - this.linkMargin;
        let get_y_position = (node_height, link_height_list, idx) => {
          let sum_height = link_height_list.sum();
          if (node_height >= sum_height) {
            let start = (node_height - sum_height) / 2;
            for (let i = 0; i < idx; ++i) start += link_height_list[i];
            return [start, start+link_height_list[idx]];
          } else {
            if (link_height_list.length === 1) return [(node_height - link_height_list[idx])/2, (node_height + link_height_list[idx])/2];
            if (sum_height > 3 * node_height) return [(node_height - link_height_list[idx])/2, (node_height + link_height_list[idx])/2];
            let overlap = (node_height - sum_height) / (link_height_list.length - 1);
            let start = 0;
            for (let i = 0; i < idx; ++i) start += link_height_list[i] + overlap;
            return [start, start+link_height_list[idx]]
            // if (link_height_list[idx] > 6) {
            //   return [start, start+link_height_list[idx]];
            // } else {
            //   return [start+link_height_list[idx]/2-3, start+link_height_list[idx]/2+3];
            // }
            
          }
        }
        let linkPath = (link) => {
          const i = link.logic_row;
          const j = link.logic_col;
          const row_links = this.links.filter(d => d.logic_row === i);
          const col_links = this.links.filter(d => d.logic_col === j);
          const [dy1, dy4] = get_y_position(logic_row_group[i].nrow * this.clusterHeight, row_links.map(d => Math.abs(d.influence)*this.influence_ratio), row_links.findIndex(d => d.logic_col === j));
          const [dy2, dy3] = get_y_position(logic_col_group[j].nrow * this.clusterHeight, col_links.map(d => Math.abs(d.influence)*this.influence_ratio * 1.2), col_links.findIndex(d => d.logic_row === i));         
          const [y1, y4] = [dy1 + logic_row_group[i].offset, dy4 + logic_row_group[i].offset];
          const [y2, y3] = [dy2 + logic_col_group[j].offset, dy3 + logic_col_group[j].offset];
          let l1 = linkH({
                  source: {
                      x: linkStart + this.linkDirect,
                      y: y1
                  },
                  target: {
                      x: linkEnd - this.linkDirect,
                      y: y2
                  }
              }).substr(1),
              l2 = linkH({
                  source: {
                      x: linkEnd - this.linkDirect,
                      y: y3
                  },
                  target: {
                      x: linkStart + this.linkDirect,
                      y: y4
                  }
              }).substr(1),
              l_core = linkH({
                  source: {
                      x: linkStart + this.linkDirect,
                      y: (y1+y4)/2
                  },
                  target: {
                      x: linkEnd - this.linkDirect,
                      y: (y2+y3)/2
                  }
                }).substr(1);
            return [`M${linkStart},${y1}L${l1}L${linkEnd},${y2}L${linkEnd},${y3}L${l2}L${linkStart},${y4}L${linkStart},${y1}Z`,
                    `M${linkStart},${(y1+y4)/2}L${l_core}L${linkEnd},${(y2+y3)/2}`];
        };

        let links = [];
        let influences = [];
        for (let i = 0; i < this.num_row_cluster; ++i) {
            for (let j = 0; j < this.num_col_cluster; ++j) {
                let row_idx = logic_row_group[i].local_indices,
                    col_idx = logic_col_group[j].local_indices;
                let influence = col_idx.map(jj => row_idx.map(ii => this.lam[ii] * this.R[ii][jj]).sum()).mean();
                // let influence = col_idx.map(jj => row_idx.map(ii => this.R[ii][jj]).sum()).mean();
                // let influence = col_idx.map(jj => row_idx.map(ii => Math.abs(this.lam[ii] * this.R[ii][jj])).sum()).mean();
                //let influence = col_idx.map(jj => row_idx.map(ii => Math.abs(this.lam[ii] * this.R[ii][jj])).sum()).sum();
                // let influence = col_idx.map(jj => row_idx.map(ii => (this.lam[ii] * this.R[ii][jj])).sum()).sum();
                links.push({
                    id: `${logic_row_group[i].id}-${logic_col_group[j].id}`,
                    row: logic_row_group[i].id,
                    col: logic_col_group[j].id,
                    logic_id: `${i}-${j}`,
                    logic_row: i,
                    logic_col: j,
                    influence: influence,
                    highlight: 0,
                    is_max: false
                });
                influences.push(Math.abs(influence));
            }
        }
        influences.sort((a,b) => a - b);
        // console.log(influences);
        const low_threshold = influences.quantile(0.1);
        let influence_ratio = 10000;
        for (let j = 0; j < this.num_col_cluster; ++j) {
          let tmp = links.filter(d => d.logic_col === j);
          let acc_influence = tmp.map(d => Math.abs(d.influence)).sum()
          tmp.forEach(d => d.col_ratio = Math.abs(d.influence) / acc_influence);
          let max_influence = tmp.map(d => Math.abs(d.influence)).max();
          tmp.find(d => Math.abs(d.influence) === max_influence).is_max = true;
          let height = logic_col_group[j].nrow * this.clusterHeight;
          let ratio = height * 1.5 / acc_influence;
          if (influence_ratio > ratio) influence_ratio = ratio;
        }
        for (let i = 0; i < this.num_row_cluster; ++i) {
          let tmp = links.filter(d => d.logic_row === i);
          let acc_influence = tmp.map(d => Math.abs(d.influence)).sum()
          tmp.forEach(d => d.row_ratio = Math.abs(d.influence) / acc_influence);
          let max_influence = tmp.map(d => Math.abs(d.influence)).max();
          tmp.find(d => Math.abs(d.influence) === max_influence).is_max = true;
          let height = logic_row_group[i].nrow * this.clusterHeight;
          let ratio = height * 1.5 / acc_influence;
          if (influence_ratio > ratio) influence_ratio = ratio;
        }
        // links = links.filter(d => d.is_max 
        // || d.col_ratio > 1 / this.num_row_cluster && Math.abs(d.influence) > low_threshold 
        // || d.row_ratio > 1 / this.num_col_cluster && Math.abs(d.influence) > low_threshold);
        links.forEach(d => d.exist = d.is_max || d.col_ratio > 1 / this.num_row_cluster && Math.abs(d.influence) > low_threshold || d.row_ratio > 1 / this.num_col_cluster && Math.abs(d.influence) > low_threshold)
        // links.forEach(d => d.exist = false)
        // links.forEach(d => d.exist = !d.exist);
        this.links = links;
        this.influence_ratio = influence_ratio * 4;
        for (let link of links) {
            [link.d, link.d_core] = linkPath(link);
        }

        this.$nextTick(function () {
          this.animateLink();
        })
        // console.log(d3.selectAll(".link-group path")._groups[0].length, this.links);
        d3.selectAll(".link-group path").data(this.links)
        this.updateLinkDisplay();

        // let links = [];
        // let link_strength = Array(this.num_row_cluster).fill(null).map(_ => Array(this.num_col_cluster).fill(null).map(_ => 0));
        // for (let i = 0; i < this.num_row_cluster; ++i) {
        //     for (let j = 0; j < this.num_col_cluster; ++j) {
        //         let row_idx = logic_row_group[i].local_indices,
        //             col_idx = logic_col_group[j].local_indices;
        //         link_strength[i][j] = col_idx.map(jj => row_idx.map(ii => this.lam[ii] * this.R[ii][jj]).sum()).mean();
        //         console.log(col_idx.map(jj => row_idx.map(ii => this.lam[ii] * this.R[ii][jj]).sum()).sum())
        //     }
        // }
        // let accumulate_strength_val = Array(this.num_row_cluster).fill(null).map((_, i) => {
        //   let tmp = link_strength[i].map(d => Math.abs(d));
        //   let ret = [];
        //   tmp.forEach((s, j) => {
        //     if (j == 0) ret.push(s); else ret.push(ret[j - 1] + s);
        //   });
        //   return ret;
        // });
        // let accumulate_strength_col = Array(this.num_col_cluster).fill(null).map((_, i) => {
        //   let tmp = link_strength.map((d, _) => Math.abs(d[i]));
        //   let ret = [];
        //   tmp.forEach((s, j) => {
        //     if (j == 0) ret.push(s); else ret.push(ret[j - 1] + s);
        //   });
        //   return ret;
        // });

        // for (let i = 0; i < this.num_row_cluster; ++i) {
        //     for (let j = 0; j < this.num_col_cluster; ++j) {
        //         links.push({
        //             id: `${logic_row_group[i].id}-${logic_col_group[j].id}`,
        //             row: logic_row_group[i].id,
        //             col: logic_col_group[j].id,
        //             logic_id: `${i}-${j}`,
        //             logic_row: i,
        //             logic_col: j,
        //             value: link_strength[i][j],
        //             row_ratio: Math.abs(link_strength[i][j]) / accumulate_strength_val[i][this.num_col_cluster - 1],
        //             col_ratio: Math.abs(link_strength[i][j]) / accumulate_strength_col[j][this.num_row_cluster - 1],
        //             d: linkPath(i, j, Math.abs(link_strength[i][j]), accumulate_strength_val[i][j], accumulate_strength_val[i][this.num_col_cluster - 1], accumulate_strength_col[j][i], accumulate_strength_col[j][this.num_row_cluster - 1]),
        //             d_core: linkCorePath(i, j, Math.abs(link_strength[i][j]), accumulate_strength_val[i][j], accumulate_strength_val[i][this.num_col_cluster - 1], accumulate_strength_col[j][i], accumulate_strength_col[j][this.num_row_cluster - 1]),
        //             highlight: 0,
        //             is_max: false
        //         });
  
        //     }
        // }
        // for (let i = 0; i < this.num_row_cluster; ++i) {
        //   let sub = links.filter(d => d.row == i);
        //   let max_ratio = sub.map(d => d.row_ratio).max();
        //   sub.find(d => d.row_ratio == max_ratio).is_max = true;
        // }
        // for (let j = 0; j < this.num_col_cluster; ++j) {
        //   let sub = links.filter(d => d.col == j);
        //   let max_ratio = sub.map(d => d.col_ratio).max();
        //   sub.find(d => d.col_ratio == max_ratio).is_max = true;
        // }
        // this.links = links;
      },
      updateValidation() {
        // selectAll('.validation-rect').attr('fill', '#666666').style("opacity", 0.6);
        // for (let idx of [...this.high_pos_validation.idx, ...this.high_neg_validation.idx]) {
        //   let cnt1 = this.high_pos_validation.count(idx);
        //   let cnt2 = this.high_neg_validation.count(idx);
        //   select('#validation-'+idx).raise().select('.validation-rect').attr('fill', cnt1 > cnt2 ? Color.GREEN_POINT : cnt1 < cnt2 ? Color.RED_POINT : "#666666").style("opacity", 1);
        // }
      },
      updateLinkDisplay() {
        console.log('update link display');
        this.links.forEach(link => link.running = false);
        this.links.forEach(link => link.display = (link.is_max || link.highlight || Math.abs(link.influence) > this.influence_threshold || link.col_ratio > this.ratio_threshold) );
        // this.links.forEach(link => link.display = false);
        // selectAll('.colored-link').raise()
        // selectAll('.colored-link-core').raise()
        this.$forceUpdate();
      },
      highlightLink(location, link) {
        this.links.forEach(link => link.display = false);
        this.links.forEach(link => link.running = false);
        if (location==='left') {
          let used_links = this.links.filter(d => d.row===link.row).filter(d => d.row_ratio>0.025);
          used_links.forEach(d => d.display = true);
          used_links.forEach(d => d.running = true);
          this.set_matrix({
            row_idx: this.row_group[link.row].indices,
            col_idx: [].concat(...used_links.map(d => this.col_group[d.col].indices))
          });
        } else if (location=='right') {
          let used_links = this.links.filter(d => d.col===link.col).filter(d => d.col_ratio>0.025);
          used_links.forEach(d => d.display = true);
          used_links.forEach(d => d.running = true);
          this.set_matrix({
            row_idx: [].concat(...used_links.map(d => this.row_group[d.row].indices)),
            col_idx: this.col_group[link.col].indices,
          });
        } else {
          link.display = true;
          link.running = true;
          this.set_matrix({
            row_idx: this.row_group[link.row].indices,
            col_idx: this.col_group[link.col].indices,
          });
        }
        this.$forceUpdate();
      },
      updateThreshold(value) {
        this.set_threshold(this.threshold)
        this.redraw_all()
      },
      updateGamma(event, value) {
        this.set_gamma(this.gamma)
        // console.log('gamma', value);
      },
      highlightLinkFromIndex() {
        this.links.forEach(link => link.display = false);
        this.links.forEach(link => link.running = false);
        let selected_row_index = this.highlight_row_index;
        let selected_col_index = this.highlight_col_index;

        let display_row_index = null, display_col_index = null;
        if (selected_row_index.length == 0 && selected_col_index.length != 0) {
          let result = [];
          for (let idx of selected_col_index) {
            result.push(this.all_samples[idx].sorted_score_list.slice(0,5));
            result.push(this.all_samples[idx].sorted_score_list.slice(-5));
          }
          display_row_index = result.flat().unique();
        } else {
          display_row_index = selected_row_index;
        }
        if (selected_col_index.length == 0 && selected_row_index.length != 0) {
          let result = [];
          for (let idx of selected_row_index) {
            result.push(this.all_samples[idx].sorted_influence_list.slice(0,5));
            result.push(this.all_samples[idx].sorted_influence_list.slice(-5));
          }
          display_col_index = result.flat().unique();

          let row = selected_row_index.map(idx => this.all_samples[idx].row_cluster).unique();
          let selected_row_local_index = selected_row_index.map(i => this.row_index.indexOf(i))
          let col_group_score = this.col_group.map(d => d.local_indices.map(jj => selected_row_local_index.map(ii => Math.abs(this.R[ii][jj])).sum()).mean());
          let top_5_col = col_group_score.map((d,i) => [d, i]).sort((a,b) => b[0]-a[0]).map(d => d[1]).slice(0, 5);
          let links = this.links.filter(d => row.includes(d.row) && top_5_col.includes(d.col));
          links = links.filter(d => d.row_ratio > 0.5/this.num_col_cluster).filter(d => d.col_ratio > 0.5/this.num_row_cluster);
          links.forEach(link => link.display = true);
          this.$forceUpdate();
          return;
        } else {
          display_col_index = selected_col_index;
        }
        let pairs = [];
        if (display_row_index.length > 0 && display_col_index.length > 0) {
          for (let r of display_row_index) {
            const used = [...this.all_samples[r].sorted_influence_list.slice(0,5), ...this.all_samples[r].sorted_influence_list.slice(0,5)];
            for (let c of display_col_index) {
              if (used.includes(c)) pairs.push([r,c]);
            }
          }
          for (let c of display_col_index) {
            const used = [...this.all_samples[c].sorted_score_list.slice(0,5), ...this.all_samples[c].sorted_score_list.slice(-5)];
            for (let r of display_row_index) {
              if (used.includes(r)) pairs.push([r,c]);
            }
          }
          for (let [r,c] of pairs) {
            let link = this.links.find(d => d.row===this.all_samples[r].row_cluster && d.col===this.all_samples[c].col_cluster);
            if (link) {
              link.display = true;
              link.running = true;
            }
          }
          this.$forceUpdate();
        } else {
          this.updateLinkDisplay();
        }


        // if (selected_row_index.length == 0 && selected_col_index.length == 0) {
        //   this.updateLinkDisplay();
        // } else if (selected_row_index.length != 0 && selected_col_index.length != 0) {
        //   let selected_row_cluster = Array.from(new Set(selected_row_index.map(idx => this.all_samples[idx].row_cluster)));
        //   let selected_col_cluster = Array.from(new Set(selected_col_index.map(idx => this.all_samples[idx].col_cluster)));
        //   let links = this.links.filter(d => selected_row_cluster.includes(d.row) && selected_col_cluster.includes(d.col));
        //   links.forEach(d => d.display = true);
        // } else {
        //   const K = 25;
        //   const N = 10;
        //   let results = [];
        //   if (selected_row_index.length != 0 && selected_col_index.length == 0) {
        //     for (let idx of selected_row_index) {
        //       let r = this.all_samples[idx].row_cluster;
        //       let tmp = [...this.all_samples[idx].sorted_influence_list.slice(0, K), ...this.all_samples[idx].sorted_influence_list.slice(-K)]
        //       for (let idx of tmp) {
        //         let c = this.all_samples[idx].col_cluster;
        //         results.push(`${r}-${c}`);
        //       }
        //     }
        //   } else if (selected_row_index.length == 0 && selected_col_index.length != 0) {
        //     for (let idx of selected_col_index) {
        //       let c = this.all_samples[idx].col_cluster;
        //       let tmp = [...this.all_samples[idx].sorted_score_list.slice(0, K), ...this.all_samples[idx].sorted_score_list.slice(-K)]
        //       for (let idx of tmp) {
        //         let r = this.all_samples[idx].row_cluster;
        //         results.push(`${r}-${c}`);
        //       }
        //     }
        //   } 
        //   const counter = results.reduce((acc, e) => acc.set(e, (acc.get(e) || 0) + 1), new Map());
        //   let values = new Array(...counter.values());
        //   let threshold = values.sort((a,b) => (a-b))[Math.max(values.length-N, 0)];
        //   for (let [key, value] of counter.entries()) {
        //     if (value >= threshold) {
        //       let [r, c] = key.split('-').map(d => parseInt(d));
              
        //       let link = this.links.find(d => d.row==r && d.col==c);
        //       console.log(key, value, r, c, link)
        //       if (link) link.display = true;
        //     }
        //   }
        // }
        // console.log(this.links.filter(d => d.display))
        // this.$forceUpdate();
        // return;
        // let selected_row_cluster = Array.from(new Set(selected_row_index.map(idx => this.all_samples[idx].row_cluster)))
        // let selected_col_cluster = Array.from(new Set(selected_col_index.map(idx => this.all_samples[idx].col_cluster)))
        // if (selected_row_cluster.length == 0 && selected_col_cluster.length == 0) {
        //   this.updateLinkDisplay();
        //   return;
        // }
        // if (selected_row_cluster.length != 0 && selected_col_cluster.length != 0) {
        //   let links = this.links.filter(d => selected_row_cluster.includes(d.row) && selected_col_cluster.includes(d.col));
        //   links.forEach(d => d.display = true);
        // }
        // if (selected_row_cluster.length == 1 && selected_col_cluster.length == 0) {
        //   let links = this.links.filter(d => selected_row_cluster.includes(d.row));
        //   links.forEach(d => d.display = true);
        // }
        // if (selected_row_cluster.length == 0 && selected_col_cluster.length == 1) {
        //   let links = this.links.filter(d => selected_col_cluster.includes(d.col));
        //   links.forEach(d => d.display = true);
        // }
        // if (selected_row_cluster.length > 1 && selected_col_cluster.length == 0) {
        //   let links = this.links.filter(d => selected_row_cluster.includes(d.row));
        //   links.forEach(d => d.display = true);
        // }
        // if (selected_row_cluster.length == 0 && selected_col_cluster.length > 1) {
        //   let links = this.links.filter(d => selected_col_cluster.includes(d.col));
        //   links.forEach(d => d.display = true);
        // }
        // this.$forceUpdate();
        // return;


        // let display_row_index = null, display_col_index = null;
        // if (selected_row_index.length == 0 && selected_col_index.length != 0) {
        //   let result = [];
        //   for (let idx of selected_col_index) {
        //     result.push(this.all_samples[idx].sorted_score_list.slice(0,5));
        //     result.push(this.all_samples[idx].sorted_score_list.slice(-5));
        //   }
        //   display_row_index = result.flat().unique();
        // } else {
        //   display_row_index = selected_row_index;
        // }
        // if (selected_col_index.length == 0 && selected_row_index.length != 0) {
        //   let result = [];
        //   for (let idx of selected_row_index) {
        //     result.push(this.all_samples[idx].sorted_influence_list.slice(0,5));
        //     result.push(this.all_samples[idx].sorted_influence_list.slice(-5));
        //   }
        //   display_col_index = result.flat().unique();
        // } else {
        //   display_col_index = selected_col_index;
        // }
        // let pairs = [];
        // if (display_row_index.length > 0 && display_col_index.length > 0) {
        //   for (let r of display_row_index) {
        //     const used = [...this.all_samples[r].sorted_influence_list.slice(0,5), ...this.all_samples[r].sorted_influence_list.slice(0,5)];
        //     for (let c of display_col_index) {
        //       if (used.includes(c)) pairs.push([r,c]);
        //     }
        //   }
        //   for (let c of display_col_index) {
        //     const used = [...this.all_samples[c].sorted_score_list.slice(0,5), ...this.all_samples[c].sorted_score_list.slice(-5)];
        //     for (let r of display_row_index) {
        //       if (used.includes(r)) pairs.push([r,c]);
        //     }
        //   }
        //   for (let [r,c] of pairs) {
        //     let link = this.links.find(d => d.row===this.all_samples[r].row_cluster && d.col===this.all_samples[c].col_cluster);
        //     if (link) {
        //       link.display = true;
        //       link.running = true;
        //     }
        //   }
        //   this.$forceUpdate();
        // } else {
        //   this.updateLinkDisplay();
        // }
      },
      linkMouseMove(link, event) {
        console.log("in");
        console.log(event, this.clusterLeftEnd, this.clusterLeftEnd + this.linkMargin + this.linkDirect, this.clusterRightStart - this.linkMargin - this.linkDirect, this.clusterRightStart)
        const x = event.offsetX - 40; // since the g element has a X-offset
        const location = x < this.clusterLeftEnd + this.linkMargin + this.linkDirect ? 'left' :
                        x > this.clusterRightStart - this.linkMargin - this.linkDirect ? 'right' : 'middle';
        if (this.linkStatus && this.linkStatus[0] === location && this.linkStatus[1].id === link.id) return;
        this.linkStatus = [location, link];
        this.highlightLink(location, link);
      },
      linkMouseOut(link, event) {
        console.log("out");
        if (this.clickStatus) {
          const [location, link] = this.clickStatus;
          this.highlightLink(location, link);
        } else {
          this.linkStatus = null;
          this.set_matrix(null);
          this.highlightLinkFromIndex();
        }
      },
      linkMouseClick(link, event) {
        if (this.clickStatus && this.clickStatus[1].id === link.id) {
          this.clickStatus = null;
        } else {
          this.clickStatus = [this.linkStatus[0], this.linkStatus[1]];
        }
      },
      // drawLegend() {
      //   const radius = 100;
      //   const g = select("#legend")
      //     .append("g")
      //     .attr("transform", `translate(${260},${75})`);
      //   const data = [
      //     {"index": 0, "pad": 0.01, "number": 20, "name": "High-confidence high-weight", color: Color.GREEN_POINT},
      //     {"index": 1, "pad": 0.01, "number": 10, "name": "High-confidence low-weight", color: Color.LIGHTGREEN},
      //     {"index": 2, "pad": 0.03, "number": 15, "name": "Low-confidence low-weight", color: Color.RED_POINT},
      //     {"index": 3, "pad": 0.03, "number": 25, "name": "Low-confidence high-weight", color: Color.LIGHTRED},
      //     {"index": 4, "pad": 0.03, "number": 60, "name": "Medium-confidence", color: "#808080"}];
      //   const arc_data = pie()
      //     .value(d => d.number)
      //     .padAngle((d,i) => 0.05)
      //     .sort((a, b) => a.index-b.index)(data);
      //   const draw_arc = arc().innerRadius(30).outerRadius(50);
      //   const label_arc = d3.arc().innerRadius(55).outerRadius(55)
      //   g.selectAll('.pie')
      //     .data(arc_data)
      //     .enter()
      //     .append('path')
      //     .classed("pie", true)
      //     .attr('d', draw_arc)
      //     .attr('fill', d => d.data.color)
      //     .style("opacity", 1);

      //   g.selectAll('.legend-allPolylines')
      //     .data(arc_data)
      //     .enter()
      //     .append('polyline')
      //     .classed('legend-allPolylines', true)
      //     .attr("stroke", "black")
      //     .style("fill", "none")
      //     .attr("stroke-width", 1)
      //     .attr('points', function(d) {
      //       let posA = draw_arc.centroid(d) // line insertion in the slice
      //       let posB = label_arc.centroid(d) // line break: we use the other arc generator that has been built only for that
      //       let posC = label_arc.centroid(d); // Label position = almost the same as posB
      //       let midangle = d.startAngle + (d.endAngle - d.startAngle) / 2 // we need the angle to see if the X position will be at the extreme right or extreme left
      //       posC[0] = radius * 0.95 * (midangle < Math.PI ? 1 : -1); // multiply by 1 or -1 to put it on the right or on the left
      //       return [posA, posB, posC]
      //     })

      //   g.selectAll('.legend-text')
      //     .data(arc_data)
      //     .enter()
      //     .append('text')
      //     .classed('legend-text', true)
      //     .text( d => d.data.name)
      //     .attr('dy', 5)
      //     .attr('transform', function(d) {
      //         let pos = label_arc.centroid(d);
      //         let midangle = d.startAngle + (d.endAngle - d.startAngle) / 2
      //         pos[0] = radius * 0.99 * (midangle < Math.PI ? 1 : -1);
      //         return 'translate(' + pos + ')';
      //     })
      //     .style('text-anchor', function(d) {
      //         let midangle = d.startAngle + (d.endAngle - d.startAngle) / 2
      //         return (midangle < Math.PI ? 'start' : 'end')
      //     })

      //   g.append('text').text(57).attr('dy',3).style('text-anchor', 'middle');
      //   g.append('text').text('Number of samples').attr('dy',5).attr('transform', 'translate(-240,35)')
      //   g.append('polyline').attr("stroke", "black")
      //     .style("fill", "none")
      //     .attr("stroke-width", 1)
      //     .attr('points', [[-92,35],[-50,35],[-10,5]]);

      //     const bar_data = [
      //       {"index": 0, "pad": 0.01, "number": 40, "name": "High-confidence high-weight", color: Color.GREEN_POINT},
      //       {"index": 1, "pad": 0.03, "number": 35, "name": "Low-confidence low-weight", color: Color.RED_POINT},
      //       {"index": 2, "pad": 0.03, "number": 25, "name": "Medium-confidence", color: "#808080"}
      //     ];
      //     const bar_data_sum = bar_data.map(d => d.number).sum();
      //     const pad = 10;
      //     const bar_width=550;
      //     g.selectAll('.legend-bar')
      //     .data(bar_data)
      //     .enter()
      //     .append('rect')
      //     .classed('legend-bar', true)
      //     .attr('width', d => d.number / bar_data_sum * bar_width)
      //     .attr('height', 10)
      //     .attr('x', (d, i) => -230 + bar_data.filter(dd => dd.index < d.index).map(dd => dd.number).sum() / bar_data_sum * bar_width + i * pad)
      //     .attr('y', 105)
      //     .attr('fill', d => d.color)
      //     .style('opacity', 1)
      //   g.selectAll('.legend-bar-text')
      //     .data(bar_data)
      //     .enter()
      //     .append('text')
      //     .classed('legend-bar-text', true)
      //     .text(d => d.name)
      //     .attr('x', (d, i) => -230 + (bar_data.filter(dd => dd.index < d.index).map(dd => dd.number).sum()+d.number/2) / bar_data_sum * bar_width + i * pad)
      //     .attr('y', 100)
      //     .attr('text-anchor', 'middle');
      // },
      // drawLegend2() {
      //   const g = select("#legend")
      //     .append("g")
      //     .attr("transform", `translate(${20},${20})`);
      //   const data = [
      //     {"shape": Shape.SQUARE, "name": "Validation samples", color: Color.DARK_GREEN},
      //     {"shape": Shape.CIRCLE, "name": "Medium-confidence training samples", color: "#808080"},
      //     {"shape": Shape.CIRCLE, "name": "High-confidence positive-weight training samples", color: Color.GREEN_POINT},
      //     {"shape": Shape.CIRCLE, "name": "High-confidence negative-weight training samples", color: Color.LIGHTGREEN},
      //     {"shape": Shape.CIRCLE, "name": "Low-confidence positive-weight training samples", color: Color.RED_POINT},
      //     {"shape": Shape.CIRCLE, "name": "Low-confidence negative-weight training samples", color: Color.LIGHTRED},
      //     ];
      //   const arc_data = pie()
      //     .value(d => d.number)
      //     .padAngle((d,i) => 0.05)
      //     .sort((a, b) => a.index-b.index)(data);
      //   const draw_arc = arc().innerRadius(30).outerRadius(50);
      //   const label_arc = d3.arc().innerRadius(55).outerRadius(55)
      //   g.selectAll('.pie')
      //     .data(arc_data)
      //     .enter()
      //     .append('path')
      //     .classed("pie", true)
      //     .attr('d', draw_arc)
      //     .attr('fill', d => d.data.color)
      //     .style("opacity", 1);

      //   g.selectAll('.legend-allPolylines')
      //     .data(arc_data)
      //     .enter()
      //     .append('polyline')
      //     .classed('legend-allPolylines', true)
      //     .attr("stroke", "black")
      //     .style("fill", "none")
      //     .attr("stroke-width", 1)
      //     .attr('points', function(d) {
      //       let posA = draw_arc.centroid(d) // line insertion in the slice
      //       let posB = label_arc.centroid(d) // line break: we use the other arc generator that has been built only for that
      //       let posC = label_arc.centroid(d); // Label position = almost the same as posB
      //       let midangle = d.startAngle + (d.endAngle - d.startAngle) / 2 // we need the angle to see if the X position will be at the extreme right or extreme left
      //       posC[0] = radius * 0.95 * (midangle < Math.PI ? 1 : -1); // multiply by 1 or -1 to put it on the right or on the left
      //       return [posA, posB, posC]
      //     })

      //   g.selectAll('.legend-text')
      //     .data(arc_data)
      //     .enter()
      //     .append('text')
      //     .classed('legend-text', true)
      //     .text( d => d.data.name)
      //     .attr('dy', 5)
      //     .attr('transform', function(d) {
      //         let pos = label_arc.centroid(d);
      //         let midangle = d.startAngle + (d.endAngle - d.startAngle) / 2
      //         pos[0] = radius * 0.99 * (midangle < Math.PI ? 1 : -1);
      //         return 'translate(' + pos + ')';
      //     })
      //     .style('text-anchor', function(d) {
      //         let midangle = d.startAngle + (d.endAngle - d.startAngle) / 2
      //         return (midangle < Math.PI ? 'start' : 'end')
      //     })

      //   g.append('text').text(57).attr('dy',3).style('text-anchor', 'middle');
      //   g.append('text').text('Number of samples').attr('dy',5).attr('transform', 'translate(-240,35)')
      //   g.append('polyline').attr("stroke", "black")
      //     .style("fill", "none")
      //     .attr("stroke-width", 1)
      //     .attr('points', [[-92,35],[-50,35],[-10,5]]);

      //     const bar_data = [
      //       {"index": 0, "pad": 0.01, "number": 40, "name": "High-confidence high-weight", color: Color.GREEN_POINT},
      //       {"index": 1, "pad": 0.03, "number": 35, "name": "Low-confidence low-weight", color: Color.RED_POINT},
      //       {"index": 2, "pad": 0.03, "number": 25, "name": "Medium-confidence", color: "#808080"}
      //     ];
      //     const bar_data_sum = bar_data.map(d => d.number).sum();
      //     const pad = 10;
      //     const bar_width=550;
      //     g.selectAll('.legend-bar')
      //     .data(bar_data)
      //     .enter()
      //     .append('rect')
      //     .classed('legend-bar', true)
      //     .attr('width', d => d.number / bar_data_sum * bar_width)
      //     .attr('height', 10)
      //     .attr('x', (d, i) => -230 + bar_data.filter(dd => dd.index < d.index).map(dd => dd.number).sum() / bar_data_sum * bar_width + i * pad)
      //     .attr('y', 105)
      //     .attr('fill', d => d.color)
      //     .style('opacity', 1)
      //   g.selectAll('.legend-bar-text')
      //     .data(bar_data)
      //     .enter()
      //     .append('text')
      //     .classed('legend-bar-text', true)
      //     .text(d => d.name)
      //     .attr('x', (d, i) => -230 + (bar_data.filter(dd => dd.index < d.index).map(dd => dd.number).sum()+d.number/2) / bar_data_sum * bar_width + i * pad)
      //     .attr('y', 100)
      //     .attr('text-anchor', 'middle');
      // },
      get_nrow(g) {
        if (g.type === 'row') return 1;
        const threshold = 1e-4;
        const constrained_number = g.indices.filter(idx => this.all_samples[idx].is_pos || this.all_samples[idx].is_neg).length;
        const constrained_ratio = constrained_number / g.indices.length;
        const unsatisfied_number = g.indices.filter(idx => this.all_samples[idx].is_pos && this.all_samples[idx].score<-threshold || this.all_samples[idx].is_neg && this.all_samples[idx].score>threshold).length;
        const unsatisfied_ratio = unsatisfied_number / g.indices.length;
        const satisfied_number = g.indices.filter(idx => this.all_samples[idx].is_pos && this.all_samples[idx].score>=threshold || this.all_samples[idx].is_neg && this.all_samples[idx].score<=-threshold).length;
        const satisfied_ratio = satisfied_number / constrained_number;

        if (this.meta_data.num_classes==14) {
          if (unsatisfied_number <= 5 || unsatisfied_ratio <= 0.15) return 0.1;
        }
        else {
          if (unsatisfied_number < 3) return 0.1;
        }
        // return 0.1;
        return 1;



        // // if ((unsatisfied_number < 3 || unsatisfied_ratio < 0.15) && constrained_ratio > 0.5) return 0.1;
        // // if ((unsatisfied_number < 3 || unsatisfied_ratio < 0.15) && constrained_ratio > 0.25) return 0.1;
        // if (unsatisfied_number<1) return 0.1;
        // if (g.indices.length==71) return 0.1;
        // // if ((unsatisfied_number < 3 || unsatisfied_ratio < 0.15) && constrained_ratio > 0.25) return 0.1;
        // if ((unsatisfied_ratio > 0.3 || unsatisfied_number > satisfied_number) && g.indices.length > 15) return 2;//Math.min(2, Math.ceil(g.indices.length/5));
        // return 1;
        // // return unsatisfied_ratio > 0.3 ? Math.min(2, Math.ceil(g.indices.length/5)) : 
        // //        satisfied_ratio > 0.8 && constrained_ratio > 0.4 ? 0.1 : 1;
      },
      buildGroup() {
        if (!this.col_index || !this.row_index) return null;
        this.col_group = Array(this.num_col_cluster).fill(0).map((_, i) => Object({
                    id: `${i}`,
                    order: i,
                    offset: 0,
                    indices: [],
                    local_indices: [],
                    sub_cluster_label: null,
                    isFocus: false,
                    nrow: 1,
                    type: 'col',
                    count: 0
                }));
        for (let group of this.col_group) {
          let cluster = this.full_col_cluster.find(cluster => cluster.id === group.id);
          group.full_index = cluster.index;
          group.display_index = cluster.display_index;
        }
        // this.col_index.forEach((idx, jj) => { 
        //     let cluster = this.all_samples[idx].col_cluster;
        //     this.col_group[cluster].indices.push(idx);
        //     this.col_group[cluster].local_indices.push(jj);
        //     this.col_group[cluster].count += 1;
        // });
        this.display_idx.forEach((jj) => { 
            let idx = this.col_index[jj]
            let cluster = this.all_samples[idx].col_cluster;
            this.col_group[cluster].indices.push(idx);
            this.col_group[cluster].local_indices.push(jj);
            this.col_group[cluster].count += 1;
        });
        let acc_offset = 0;
        this.col_group.forEach((g, j) => {
            g.weight = g.local_indices.map(jj => this.lam.map((l, ii) => l * this.R[ii][jj]).sum()).mean()
            let ReLu = (x) => Math.max(0, x);
            g.actualWeight = ReLu(g.weight) / this.col_group.map(gg => ReLu(gg.weight)).sum(); 
            g.color = this.weightColor(g.weight);
            g.indices.sort((idx1, idx2) => this.all_samples[idx1].col_x - this.all_samples[idx2].col_x);
            g.local_indices.sort((idx1, idx2) => this.all_samples[this.col_index[idx1]].col_x - this.all_samples[this.col_index[idx2]].col_x);
            g.x = g.indices.map(idx => this.all_samples[idx].col_x);
            g.w = g.indices.map(idx => this.all_samples[idx].score);
            g.full_x = g.full_index.map(idx => this.all_samples[idx].col_pos);
            g.full_w = g.full_index.map(idx => this.all_samples[idx].full_score);
            g.w_ = g.indices.map(idx => this.all_samples[idx].score_);
            g.sub_cluster_label = g.indices.map(idx => this.all_samples[idx].col_sub_cluster_label);
            g.sub_cluster_count = Math.max.apply(null, g.sub_cluster_label) + 1;
            g.nrow = this.get_nrow(g);
            g.offset = acc_offset;
            acc_offset += g.nrow * this.clusterHeight + this.clusterMargin;
        })
        this.row_group = Array(this.num_row_cluster).fill(0).map((_, i) => Object({
                    id: i,
                    order: i,
                    offset: i * (this.clusterHeight + this.clusterMargin) + (this.num_col_cluster - this.num_row_cluster) * this.clusterHeight,
                    indices: [],
                    local_indices: [],
                    sub_cluster_label: null,
                    isFocus: false,
                    nrow: 1,
                    type: 'row',
                    count: 0
                }));
        this.row_index.forEach((idx, ii) => {
            let row_cluster = this.all_samples[idx].row_cluster;
            this.row_group[row_cluster].indices.push(idx);
            this.row_group[row_cluster].local_indices.push(ii);
            this.row_group[row_cluster].count += 1;
        });
        this.row_group.forEach((g, i) => {
            g.weight = g.local_indices.map(ii => this.lam[ii]).sum();
            g.color = this.weightColor(g.weight);
            g.indices.sort((idx1, idx2) => this.all_samples[idx1].row_x - this.all_samples[idx1].row_x);
            g.local_indices.sort((idx1, idx2) => this.all_samples[this.row_index[idx1]].row_x - this.all_samples[this.row_index[idx2]].row_x);
            g.x = g.indices.map(idx => this.all_samples[idx].row_x);
            g.w = g.indices.map(idx => this.all_samples[idx].lam);
            g.w_ = g.indices.map(idx => this.all_samples[idx].lam_);
            g.full_x = g.indices.map(idx => this.all_samples[idx].row_x);
            g.full_w = g.indices.map(idx => this.all_samples[idx].lam);
            g.sub_cluster_label = g.indices.map(idx => this.all_samples[idx].row_sub_cluster_label);
            g.sub_cluster_count = Math.max.apply(null, g.sub_cluster_label) + 1;
        })
        this.updateGroupHeight();
        this.updateLinks();
        this.reorderGroup();
        this.updateGroupHeight();
        this.updateLinks();
      },
      reorderGroup() {
        // const labels1 = this.row_group.map(d => ({'id': `row-${d.id}`}));
        // const labels2 = this.col_group.map(d => ({'id': `col-${d.id}`}));
        // const graph2 = reorder.mat2graph([[1,1,1,0],[0,0,0,1],[1,1,0,0]], true);
        // console.log(graph2);
        // const labels1 = this.row_group.map(d => ({'id': d.id}));
        // const labels2 = this.col_group.map(d => ({'id': d.id + this.row_group.length}));
        // const nodes = [...labels1, ...labels2];
        // // const edges = this.links.filter(d => d.display).map(d => ({source:`row-${d.row}`, target: `col-${d.col}`}))
        // const edges = this.links.filter(d => d.display).map(d => ({source:d.row, target: d.col+this.row_group.length}))
        // const graph = reorder.graph(nodes, edges, false).init();
        // console.log(graph)
        if (!this.reorder) {
          let mat = new Array(this.row_group.length).fill(null).map(() => new Array(this.col_group.length).fill(0));
          this.links.filter(d => d.display).forEach(d => mat[d.row][d.col] = 1);
          const graph = reorder.mat2graph(mat, true);
          // const initial_crossings = reorder.count_crossings(graph);
          // this.reorder = reorder.barycenter_order(graph);
          const barycenter = reorder.barycenter_order(graph);
          this.reorder = reorder.adjacent_exchange(graph, ...barycenter); 
        }
        this.reorder[1].forEach((order, i) => this.row_group[order].order = i);
        this.reorder[0].forEach((order, i) => this.col_group[order].order = i);
        // this.row_group.forEach((d, i) => d.order = perms[1][i]);
        // this.col_group.forEach((d, i) => d.order = perms[0][i]);
        // const perms2 = reorder.adjacent_exchange(graph, perms[0], perms[1]);
        // this.row_group = perms[1].map(i => this.row_group[i]);
        // this.row_group.forEach((d,i) => d.id = i);
        // this.all_samples.filter(d => d && d.row_cluster != undefined).forEach(d => d.row_cluster = perms[1][d.row_cluster])
        // this.all_samples.filter(d => d && d.row_cluster != undefined).forEach(d => d.row_cluster = perms[1].indexOf(d.row_cluster))
        // this.col_group = perms[0].map(i => this.col_group[i]);
        // this.col_group.forEach((d,i) => d.id = i);
        // this.all_samples.filter(d => d && d.col_cluster != undefined).forEach(d => d.col_cluster = perms[0][d.col_cluster])
        // this.all_samples.filter(d => d && d.col_cluster != undefined).forEach(d => d.col_cluster = perms[0].indexOf(d.col_cluster))
      },
      animateLink() {
        // Animation
        // selectAll(".link-group path.link")
        //   .data(this.links)
        //   .transition()
        //   .duration(1000)
        //   .attr("d", link => link.d);
        // selectAll(".link-group path.bglink")
        //   .data(this.links, d=>d.id)
        //   .transition()
        //   .duration(1000)
        //   .attr("d", link => link.d);
        // selectAll(".link-core-group path")
        //   .data(this.links)
        //   .transition()
        //   .duration(1000)
        //   .attr("d", link => link.d_core);
      },
      animateGroupTitle() {
        this.rowGroupTranslate = this.row_group ? this.row_group.map(d => d.offset).min()-30 : 0;
        this.colGroupTranslate = this.col_group ? this.col_group.map(d => d.offset).min()-30 : 0;
        select("#row-group-title")
          .transition()
          .duration(1000)
          .attr("y", this.rowGroupTranslate+5);
        select("#col-group-title")
          .transition()
          .duration(1000)
          .attr("y", this.colGroupTranslate+5);
      },
      onWheel(event) {
        console.log(event.deltaY);
        if (event.deltaY > 0) {
          this.scrollTranslate += 15;
        } else {
          this.scrollTranslate -= 15;
        }
        if (this.scrollTranslate < 0) {
          this.scrollTranslate = 0;
        } else if (this.scrollTranslate > this.viewHeight - 50) {
          this.scrollTranslate = this.viewHeight - 50;
        }
        select("#bipartite-group")
          .attr("transform", `translate(75, ${this.bipartiteTransalate - this.scrollTranslate})`);
      }
    },
    computed: {
      ...mapState(['meta_data','all_samples', 'row_index', 'col_index', 'display_idx', 'num_col_cluster', 'num_row_cluster', 'lam', 'R', 'opt_result', 'highlight_row_index', 'highlight_col_index', 'legend','redraw_cnt', 'lam_type','matrix', 'full_col_cluster']),
    },
    update() {
      this.updateLinks();
    },
    watch: {
      redraw_cnt: function() {
        this.buildGroup();
        // this.updateLinks();
        this.$forceUpdate();
        console.log("link elem", d3.selectAll(".link-group path")._groups[0].length);
      },
      // bipartiteTransalate: function(newValue) {
      //   gsap.to(this.$data, { duration: 0.5, tweenedNumber: newValue });
      // }
    }
  });
</script>

<style scoped>
@keyframes dash {
    to {
        stroke-dashoffset: 500;
    }
}
.link {
  stroke-width: 0;
}
/* .link.running {
  stroke-dasharray: 30 8;
  animation: dash 15s linear;
  stroke-width: 1;
} */
</style>

<style scoped>

.fade-enter-active, .fade-leave-active {
  transition: opacity 1s;
}
.fade-enter, .fade-leave-to /* .fade-leave-active below version 2.1.8 */ {
  opacity: 0;
}
rect.selection {
  fill: "#bbb";
}
.invisible-link{
  display: none
}

</style>