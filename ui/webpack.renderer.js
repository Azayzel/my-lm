const path = require("path");

module.exports = {
  mode: "development",
  entry: "./src/renderer/app.ts",
  target: "electron-renderer",
  module: {
    rules: [
      {
        test: /\.ts$/,
        use: {
          loader: "ts-loader",
          options: {
            configFile: "tsconfig.renderer.json",
          },
        },
        exclude: /node_modules/,
      },
    ],
  },
  resolve: {
    extensions: [".ts", ".js"],
  },
  output: {
    filename: "app.js",
    path: path.resolve(__dirname, "dist/renderer"),
  },
  devtool: "source-map",
};
