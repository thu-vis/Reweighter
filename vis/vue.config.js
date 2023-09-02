module.exports = {
  "transpileDependencies": [
    "vuetify"
  ],
  "devServer": {
    host: '0.0.0.0',
    port: 49623,
    proxy: {
      '/api': {
        target: 'http://localhost:49621',//设置你调用的接口域名和端口号 别忘了加http
        changeOrigin: true,//如果需要跨域
        pathRewrite: {
          '^/api': '',//调用接口直接写‘/api/user/add’即可
        },
      },
      '/modelapi': {
        target: 'http://localhost:49622',//设置你调用的接口域名和端口号 别忘了加http
        changeOrigin: true,//如果需要跨域
        pathRewrite: {
          '^/modelapi': '',//调用接口直接写‘/api/user/add’即可
        },
      }
    }
  }
}
