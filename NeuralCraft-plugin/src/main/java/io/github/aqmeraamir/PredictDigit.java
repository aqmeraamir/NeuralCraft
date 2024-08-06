package io.github.aqmeraamir;

import org.bukkit.*;
import org.bukkit.block.Block;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.command.TabExecutor;
import org.bukkit.entity.Player;
import org.bukkit.event.Listener;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class PredictDigit implements CommandExecutor, TabExecutor {
    private final Listener blockPlaceListener;

    public PredictDigit(Listener blockPlaceListener) {
        this.blockPlaceListener = blockPlaceListener;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        // Check if any arguments are given
        if (args.length > 0) {

            // Predict command
            if (args[0].equals("predict")) {
                // Fetch the world
                World world = Bukkit.getWorld("world");
                assert world != null;

                int[] blocks_array = new int[784];
                int i = 0;
                // Iterate through each of the possible placed blocks coordinates
                for (int z = 0; z<28; z++) {
                    for (int x = 0; x<28; x++) {
                        // Fetch the specific block of the coordinates
                        Block block = world.getBlockAt(x, 110, z);

                        // If there is a placed block, set the index to 1 in the array
                        if (block.getType() != Material.AIR)  {
                            blocks_array[i] = 1;
                        }

                        i++;
                    }
                }

                // Format the blocks array, so it can be read by the API
                String formatted_array = Arrays.toString(blocks_array);
                formatted_array = formatted_array.substring(1, formatted_array.length() - 1);

                // Make an API request to retrieve a prediction of the drawn image
                try {

                    URL url = new URL("http://127.0.0.1:8080/predict_digit");

                    // open a connection
                    HttpURLConnection conn = (HttpURLConnection) url.openConnection();

                    // set a request method
                    conn.setRequestMethod("POST");
                    conn.setDoOutput(true);
                    conn.setRequestProperty("Content-Type", "application/json");

                    // create the json payload
                    String jsonPayLoad = "{\"array\":\"" + formatted_array + "\"}";

                    // write the payload to output stream
                    try (OutputStream os = conn.getOutputStream()) {
                        byte[] input = jsonPayLoad.getBytes(StandardCharsets.UTF_8);
                        os.write(input, 0, input.length);
                    }

                    // get the response code
                    int responseCode = conn.getResponseCode();
                    System.out.println("Response Code: " + responseCode);

                    // read the response using a string builder
                    BufferedReader in = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                    String inputLine;
                    StringBuilder response = new StringBuilder();

                    while ((inputLine = in.readLine()) != null) {
                        response.append(inputLine);
                    }
                    in.close();

                    // Print the HTTP response
                    System.out.println("Response: " + response);

                    // retrieve the 'most_likely' value from the string
                    int mostLikelyIndex = response.indexOf("\"most_likely\"");
                    int startIndex = response.indexOf(":", mostLikelyIndex) + 1;

                    int endIndex = response.indexOf(",", startIndex);
                    if (endIndex == -1) { // Handle the case when it's the last key-value pair
                        endIndex = response.indexOf("}", startIndex);
                    }

                    // Extract the value and trim any surrounding spaces or quotes
                    String mostLikely = response.substring(startIndex, endIndex).replaceAll("[\"\\s]", "");

                    Bukkit.broadcastMessage(ChatColor.BOLD + "The digit you drew is most likely: " + ChatColor.GOLD + mostLikely + ChatColor.GRAY + "\ntype '/digit canvas' to reset the canvas");

                } catch (Exception e) {
                    System.out.println("Error fetching prediction from the API (is it off?)");
                }
            }

            // Reset command
            if (args[0].equals("canvas")) {

                // Check if the sender is a player
                if (!(sender instanceof Player)) {
                    sender.sendMessage("This command can only be used by players.");
                    return true;
                }
                Player player = (Player) sender;

                // Commands to create the canvas
                Bukkit.dispatchCommand(Bukkit.getConsoleSender(), "fill 0 109 0 27 109 27 minecraft:white_concrete");
                Bukkit.dispatchCommand(Bukkit.getConsoleSender(), "fill 0 110 0 27 110 27 minecraft:air");

                Bukkit.dispatchCommand(Bukkit.getConsoleSender(), "fill -1 109 -1 28 109 -1 minecraft:red_concrete");
                Bukkit.dispatchCommand(Bukkit.getConsoleSender(), "fill -1 109 -1 -1 109 28 minecraft:red_concrete");
                Bukkit.dispatchCommand(Bukkit.getConsoleSender(), "fill 28 109 28 28 109 -1 minecraft:red_concrete");
                Bukkit.dispatchCommand(Bukkit.getConsoleSender(), "fill 28 109 28 -1 109 28 minecraft:red_concrete");

                Bukkit.dispatchCommand(Bukkit.getConsoleSender(), "setblock 13 110 -1 minecraft:oak_sign");
                Bukkit.dispatchCommand(Bukkit.getConsoleSender(), "setblock 14 110 -1 minecraft:oak_sign");
                Bukkit.dispatchCommand(Bukkit.getConsoleSender(), "data modify block 13 110 -1 front_text.messages set value ['\"\"', '\"Top\"', '\"\"', '\"\"']");
                Bukkit.dispatchCommand(Bukkit.getConsoleSender(), "data modify block 14 110 -1 front_text.messages set value ['\"\"', '\"Side\"', '\"\"', '\"\"']");

                // Teleport the player to the canvas
                player.teleport(new Location(player.getWorld(), 14, 111, 14, 180, 0));

                // Prompt
                Bukkit.broadcastMessage("--------------------\n" + ChatColor.GREEN + "The canvas has been built\n"  + ChatColor.GRAY + "- Build only on top of the white area, and in the right orientation following the signs\n- Run '/digit predict' when finished\n" + ChatColor.WHITE + "--------------------");

                // Enable the event listener
                //((EventListener) blockPlaceListener).setListening(true);
            }

        }

        else {
            sender.sendMessage(ChatColor.RED + "Incorrect Usage:");
            return false;
        }

        return true;
    }

    @Override
    public List<String> onTabComplete(CommandSender commandSender, Command command, String label, String[] args) {
        if (args.length == 1) {
            return Arrays.asList("predict", "canvas");
        }
        return new ArrayList<>();
    }
}


