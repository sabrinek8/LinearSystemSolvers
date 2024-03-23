import java.util.Random;
import java.util.TreeMap;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.*;
import java.util.HashMap;
import java.util.Map;

public class GvGS {

    public static void main(String[] args) {
        int[] sizes = {100, 400, 500, 700, 1000, 1500, 2000};
        int numTrials = 5;

        Map<Integer, Double> gaussTimes = new HashMap<>();
        Map<Integer, Double> gsTimes = new HashMap<>();

        for (int n : sizes) {
            double gaussTotalTime = 0;
            double gsTotalTime = 0;

            for (int t = 0; t < numTrials; t++) {
                double[][] A = generateRandomMatrix(n);
                double[] b = generateRandomVector(n);

                long startTime = System.nanoTime();
                gaussElimination(A.clone(), b.clone());
                gaussTotalTime += (System.nanoTime() - startTime) / 1e9;

                startTime = System.nanoTime();
                gaussSeidel(A.clone(), b.clone());
                gsTotalTime += (System.nanoTime() - startTime) / 1e9;
            }

            double gaussAvgTime = gaussTotalTime / numTrials;
            double gsAvgTime = gsTotalTime / numTrials;

            gaussTimes.put(n, gaussAvgTime);
            gsTimes.put(n, gsAvgTime);
        }

        SwingUtilities.invokeLater(() -> createAndShowGUI(gaussTimes, gsTimes));
    }

    private static void createAndShowGUI(Map<Integer, Double> gaussTimes, Map<Integer, Double> gsTimes) {
        JFrame frame = new JFrame("Execution Time Comparison");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        JPanel panel = new ChartPanel(gaussTimes, gsTimes);
        frame.add(panel, BorderLayout.CENTER);

        frame.pack();
        frame.setVisible(true);
    }

    static class ChartPanel extends JPanel {
        private Map<Integer, Double> gaussTimes;
        private Map<Integer, Double> gsTimes;

        public ChartPanel(Map<Integer, Double> gaussTimes, Map<Integer, Double> gsTimes) {
            this.gaussTimes = new TreeMap<>(gaussTimes);
            this.gsTimes = new TreeMap<>(gsTimes);
            setPreferredSize(new Dimension(800, 600));
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g;

            int padding = 50;
            int maxValue = (int) Math.ceil(Math.max(getMaxValue(gaussTimes), getMaxValue(gsTimes)));

            int xScale = (getWidth() - 2 * padding) / (gaussTimes.size() - 1);
            int yScale = (getHeight() - 2 * padding) / maxValue;

            // Draw x-axis
            g2.drawLine(padding, getHeight() - padding, getWidth() - padding, getHeight() - padding);

            // Draw y-axis
            g2.drawLine(padding, getHeight() - padding, padding, padding);

            // Draw x-axis labels
            int index = 0;
            for (int key : gaussTimes.keySet()) {
                int x = padding + index * xScale;
                g2.drawString(Integer.toString(key), x, getHeight() - padding / 2);
                index++;
            }

            // Draw y-axis labels
            for (int i = 0; i <= maxValue; i++) {
                int y = getHeight() - padding - i * yScale;
                g2.drawString(Integer.toString(i), padding / 2, y);
            }

            // Draw Gauss times
            drawChart(g2, gaussTimes, xScale, yScale, padding, Color.RED, "Gauss Elimination");

            // Draw Gauss-Seidel times
            drawChart(g2, gsTimes, xScale, yScale, padding, Color.BLUE, "Gauss-Seidel");
        }

        private void drawChart(Graphics2D g2, Map<Integer, Double> times, int xScale, int yScale, int padding, Color color, String label) {
            int index = 0;
            double prevY = 0;
            for (double time : times.values()) {
                int x = padding + index * xScale;
                int y = getHeight() - padding - (int) (time * yScale);
                if (index != 0) {
                    g2.draw(new Line2D.Double(padding + (index - 1) * xScale, prevY, x, y));
                }
                g2.setColor(color);
                g2.fill(new Ellipse2D.Double(x - 2, y - 2, 4, 4));
                prevY = y;
                index++;
            }

            // Draw legend
            g2.setColor(color);
            g2.fill(new Rectangle2D.Double(padding, times == gaussTimes ? 0 : 25, 20, 20));
            g2.setColor(Color.BLACK);
            g2.drawString(label, padding + 25, times == gaussTimes ? 15 : 40);
        }


        private int getMaxValue(Map<Integer, Double> map) {
            return (int) Math.ceil(map.values().stream().mapToDouble(Double::doubleValue).max().orElse(0));
        }
    }

    public static double[][] generateRandomMatrix(int n) {
        double[][] A = new double[n][n];
        Random random = new Random();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = random.nextDouble();
            }
        }
        return A;
    }

    public static double[] generateRandomVector(int n) {
        double[] b = new double[n];
        Random random = new Random();
        for (int i = 0; i < n; i++) {
            b[i] = random.nextDouble();
        }
        return b;
    }

    public static double[] gaussElimination(double[][] A, double[] b) {
        int n = b.length;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double factor = A[j][i] / A[i][i];
                for (int k = i; k < n; k++) {
                    A[j][k] -= factor * A[i][k];
                }
                b[j] -= factor * b[i];
            }
        }

        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            x[i] = (b[i] - dotProduct(A[i], x, i + 1, n)) / A[i][i];
        }

        return x;
    }

    public static double[] gaussSeidel(double[][] A, double[] b) {
        int n = b.length;
        double[] x = new double[n];
        double tol = 1e-6;
        int maxIter = 1000;

        for (int iter = 0; iter < maxIter; iter++) {
            double[] xNew = new double[n];
            for (int i = 0; i < n; i++) {
            	xNew[i] = (b[i] - dotProduct(A[i], x, 0, i) - dotProduct(A[i], x, i + 1, n)) / A[i][i];
                        xNew[i] = (b[i] - dotProduct(A[i], x, 0, i) - dotProduct(A[i], x, i + 1, n)) / A[i][i];
                    }
                    if (norm(subtract(xNew, x)) < tol) {
                        return xNew;
                    }
                    x = xNew;
                }
                return x;
            }

            public static double dotProduct(double[] a, double[] b, int start, int end) {
                double result = 0;
                for (int i = start; i < end; i++) {
                    result += a[i] * b[i];
                }
                return result;
            }

            public static double[] subtract(double[] a, double[] b) {
                double[] result = new double[a.length];
                for (int i = 0; i < a.length; i++) {
                    result[i] = a[i] - b[i];
                }
                return result;
            }

            public static double norm(double[] a) {
                double result = 0;
                for (double value : a) {
                    result += value * value;
                }
                return Math.sqrt(result);
            }
        }
